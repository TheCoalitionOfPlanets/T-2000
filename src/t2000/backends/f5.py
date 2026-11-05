"""Direct reference-conditioned F5 synthesis in a persistent isolated process."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np
import soundfile as sf

from .base import AudioChunk
from ..errors import T2000Error


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class F5Backend:
    requires_reference_text = True
    max_reference_seconds = 12

    def __init__(self, model_dir: Path, *, python_executable: Path, device="cuda:0", num_threads=4, steps=32):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.bundle_dir = self.model_dir / "f5"
        # Resolving the interpreter symlink would bypass its virtual environment.
        self.python_executable = Path(python_executable).expanduser().absolute()
        self.device, self.num_threads, self.steps = device, num_threads, steps
        if not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None:
            raise T2000Error("Use cpu, cuda, or cuda:N for the F5 device.")
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or not 1 <= num_threads <= 32:
            raise T2000Error("F5 CPU threads must be an integer between 1 and 32.")
        if isinstance(steps, bool) or not isinstance(steps, int) or not 8 <= steps <= 64:
            raise T2000Error("F5 sampling steps must be an integer between 8 and 64.")
        self._project = Path(__file__).resolve().parents[3]
        self._worker_script = self._project / "scripts/f5_infer_worker.py"
        self._process = None
        self._stderr = None
        self._lock = threading.RLock()
        self._loaded_signature = None
        self._identity_stamp = None
        self._identity_cache = None
        self._versions = None
        self._last_metrics = None

    def _identity(self):
        try:
            if not self.python_executable.is_file():
                raise ValueError("The isolated .venv-f5 Python interpreter is missing")
            manifest = json.loads((self.bundle_dir / "manifest.json").read_text())
            if (not isinstance(manifest, dict) or manifest.get("schema") != 1
                    or manifest.get("complete") is not True
                    or not isinstance(manifest.get("files"), dict) or not manifest["files"]
                    or not isinstance(manifest.get("sources"), dict) or not manifest["sources"]):
                raise ValueError("Incomplete F5 model manifest")
            # Once verified, stat the checker's complete file set rather than relying
            # on a manifest that could omit a changed source or checkpoint.
            checked = self._identity_cache or manifest
            names = sorted(set(checked["files"]) | set(checked["sources"]))
            if any(not isinstance(name, str) for name in names):
                raise ValueError("Invalid F5 manifest file names")
            stamp = []
            for name in names:
                path = (self.bundle_dir / name).resolve()
                if not path.is_relative_to(self.bundle_dir):
                    raise ValueError("F5 manifest path escapes its model directory")
                stat = path.stat()
                stamp.append((name, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
            stamp.append(("manifest", sha256(self.bundle_dir / "manifest.json")))
            if stamp != self._identity_stamp:
                path = self._project / "scripts/setup_f5.py"
                spec = importlib.util.spec_from_file_location("_t2000_f5_setup", path)
                setup = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(setup)
                self._identity_cache = setup.check_bundle(self.bundle_dir)
                if manifest != self._identity_cache:
                    raise ValueError("F5 manifest does not match its pinned bundle")
                self._identity_stamp = stamp
            if self._versions is None:
                command = (
                    "import importlib.metadata as m,json; "
                    "print(json.dumps({n:m.version(n) for n in "
                    "['torch','torchaudio','numpy','vocos','x-transformers','torchdiffeq','safetensors']}))"
                )
                result = subprocess.run([str(self.python_executable), "-c", command],
                                        capture_output=True, text=True, timeout=30, check=True)
                self._versions = json.loads(result.stdout)
            return {"bundle": self._identity_cache, "versions": self._versions,
                    "worker_sha256": sha256(self._worker_script), "adapter_sha256": sha256(Path(__file__)),
                    "python": str(self.python_executable), "device": self.device,
                    "num_threads": self.num_threads, "steps": self.steps}
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            raise T2000Error(f"F5 models or isolated dependencies are unavailable: {exc}. "
                             "Install requirements/inference-f5.txt in .venv-f5 and run scripts/setup_f5.py.") from exc

    def signature(self):
        return hashlib.sha256(json.dumps(self._identity(), sort_keys=True).encode()).hexdigest()

    def _diagnostics(self):
        """Why the worker died: its exit status and the tail of its stderr.

        Read before close() discards the capture file, so an out-of-memory kill
        is reported as one instead of as a transport error.
        """
        details = []
        status = None if self._process is None else self._process.poll()
        if status is not None and status < 0:
            try:
                name = signal.Signals(-status).name
            except ValueError:
                name = f"signal {-status}"
            details.append(f"the worker was killed by {name}" + (
                "; the host ran out of memory loading the model" if -status == signal.SIGKILL else ""))
        elif status:
            details.append(f"the worker exited with status {status}")
        if self._stderr is not None:
            try:
                size = self._stderr.seek(0, os.SEEK_END)
                self._stderr.seek(max(0, size - 8192))
                tail = self._stderr.read().decode("utf-8", "replace")
            except (OSError, ValueError):
                tail = ""
            lines = [line.strip() for line in tail.splitlines() if line.strip()][-3:]
            if lines:
                details.append("worker output: " + " | ".join(lines))
        return "; ".join(details)

    def _exchange(self, request, timeout=600):
        if self._process is None or self._process.poll() is not None:
            raise T2000Error("The F5 worker stopped. Prepare the voice again or restart the studio.")
        identity = uuid.uuid4().hex
        try:
            self._process.stdin.write(json.dumps(dict(request, request_id=identity)) + "\n")
            self._process.stdin.flush()
            deadline = time.monotonic() + timeout
            received = bytearray()
            while b"\n" not in received:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("F5 inference exceeded its response deadline")
                ready, _, _ = select.select([self._process.stdout], [], [], remaining)
                if not ready:
                    raise TimeoutError("F5 inference exceeded its response deadline")
                block = os.read(self._process.stdout.fileno(), 65536)
                if not block:
                    raise RuntimeError("F5 worker exited before returning audio")
                received.extend(block)
                if len(received) > 1024 * 1024:
                    raise ValueError("F5 worker response exceeds 1 MiB")
            response = json.loads(received.decode("utf-8"))
            if not isinstance(response, dict):
                raise ValueError("F5 worker response must be an object")
            if response.get("request_id") != identity:
                raise ValueError("F5 worker response ID does not match this request")
            if not response.get("ok"):
                raise T2000Error(response.get("error", "F5 generation failed"))
            return response
        except T2000Error:
            raise
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            details = self._diagnostics()
            self.close()
            raise T2000Error(f"F5 generation failed: {exc}"
                             + (f" ({details})" if details else "") + ".") from exc

    def load(self):
        with self._lock:
            signature = self.signature()
            if self._process is not None and self._process.poll() is None:
                if self._loaded_signature != signature:
                    raise T2000Error("F5 files or settings changed after load. Restart before generating.")
                return
            self.close()
            self._stderr = tempfile.TemporaryFile()
            try:
                self._process = subprocess.Popen(
                    [str(self.python_executable), str(self._worker_script), "--model-dir", str(self.bundle_dir),
                     "--device", self.device, "--num-threads", str(self.num_threads)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                    text=True, bufsize=1, cwd=self._project,
                )
                self._exchange({"command": "load"})
                self._loaded_signature = signature
            except OSError as exc:
                details = self._diagnostics()
                self.close()
                raise T2000Error(f"Could not start the isolated F5 worker: {exc}"
                                 + (f" ({details})" if details else "") + ".") from exc
            except BaseException:
                self.close()
                raise

    def prepare(self, reference_path: Path, cache_dir: Path, *, reference_text=""):
        if not isinstance(reference_text, str) or not reference_text.strip() or len(reference_text) > 2000:
            raise T2000Error("F5 needs the exact reference transcript. Transcribe the recording and check its words.")
        reference_path = Path(reference_path).expanduser().resolve()
        try:
            info = sf.info(reference_path)
        except (OSError, RuntimeError) as exc:
            raise T2000Error(f"Could not read the F5 reference recording: {exc}") from exc
        if not 1 <= info.duration <= self.max_reference_seconds:
            raise T2000Error("Use a complete 1–12 second F5 reference with its matching transcript.")
        with self._lock:
            self.load()
            value = {"schema": 1, "reference_path": str(reference_path),
                     "reference_sha256": sha256(reference_path), "reference_text": reference_text.strip(),
                     "signature": self._loaded_signature, "duration_seconds": info.duration}
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            destination = cache_dir / "f5-reference.json"
            if destination.is_file():
                try:
                    if json.loads(destination.read_text()) == value:
                        return
                except (OSError, ValueError):
                    pass
            fd, temporary = tempfile.mkstemp(prefix=".f5-reference-", suffix=".json.tmp", dir=cache_dir)
            try:
                with os.fdopen(fd, "w") as output:
                    output.write(json.dumps(value, indent=2) + "\n")
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)

    def synthesize(self, text, cache_dir, *, speed, seed):
        if not isinstance(text, str) or not text.strip() or len(text) > 600:
            raise T2000Error("F5 needs a nonempty phrase of at most 600 characters.")
        if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                or not math.isfinite(speed) or not 0.5 <= speed <= 2):
            raise T2000Error("F5 speed must be a finite number between 0.5 and 2.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise T2000Error("F5 seed must be an integer between 0 and 4294967295.")
        with self._lock:
            self.load()
            try:
                profile = json.loads((Path(cache_dir) / "f5-reference.json").read_text())
                if (not isinstance(profile, dict) or profile.get("schema") != 1
                        or not isinstance(profile.get("reference_text"), str)
                        or not profile["reference_text"].strip() or len(profile["reference_text"]) > 2000
                        or not isinstance(profile.get("reference_path"), str)
                        or profile["signature"] != self._loaded_signature
                        or sha256(profile["reference_path"]) != profile["reference_sha256"]):
                    raise ValueError("reference or F5 signature changed")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise T2000Error(f"F5 reference conditioning is unavailable ({exc}). Prepare this voice again.") from exc
            with tempfile.TemporaryDirectory(prefix="f5-output-", dir=cache_dir) as temporary:
                output_path = Path(temporary) / "speech.wav"
                result = self._exchange({"reference_path": profile["reference_path"],
                                         "reference_text": profile["reference_text"], "text": text,
                                         "output_path": str(output_path.resolve()), "seed": seed,
                                         "speed": speed, "steps": self.steps})
                clipping = result.get("clipped_fraction")
                generation_seconds = result.get("generation_seconds")
                if (isinstance(clipping, bool) or not isinstance(clipping, (int, float))
                        or not math.isfinite(clipping) or clipping < 0
                        or isinstance(generation_seconds, bool) or not isinstance(generation_seconds, (int, float))
                        or not math.isfinite(generation_seconds) or generation_seconds < 0):
                    raise T2000Error("F5 worker returned invalid audio metrics.")
                if clipping > 0:
                    raise T2000Error("F5 produced clipped audio. Try another clean reference or seed.")
                output_info = sf.info(output_path)
                if output_info.channels != 1 or output_info.samplerate != 24000 or not 0 < output_info.duration <= 600:
                    raise T2000Error("F5 produced invalid audio dimensions or duration.")
                samples, rate = sf.read(output_path, dtype="float32", always_2d=True)
                if (rate != 24000 or samples.shape[1] != 1 or not samples.size
                        or not np.isfinite(samples).all() or not np.any(samples)):
                    raise T2000Error("F5 produced invalid audio.")
                if np.any(np.abs(samples) >= 32767 / 32768):
                    raise T2000Error("F5 produced clipped audio. Try another clean reference or seed.")
                self._last_metrics = {key: result.get(key) for key in
                                      ("cuda_memory", "chunks", "raw_peak", "raw_clipped_fraction", "output_gain")}
                return AudioChunk(np.ascontiguousarray(samples[:, 0]), rate, generation_seconds, 0.0)

    def info(self):
        loaded = self._process is not None and self._process.poll() is None and self._loaded_signature is not None
        return {"backend": "f5-tts-v1", "loaded": loaded, "model_dir": str(self.bundle_dir),
                "tts_device": self.device, "converter_device": None, "sample_rate": 24000, "steps": self.steps,
                "reference_mode": "direct audio and transcript", "signature": self._loaded_signature,
                "streaming": "complete phrases", "requires_reference_text": True,
                "max_reference_seconds": self.max_reference_seconds, "last_worker": self._last_metrics}

    def close(self):
        with self._lock:
            process, self._process = self._process, None
            self._loaded_signature = None
            if process is not None:
                if process.poll() is None:
                    try:
                        process.stdin.write('{"command":"close"}\n')
                        process.stdin.flush()
                        process.wait(timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                for stream in (process.stdin, process.stdout):
                    if stream:
                        # A dead peer can still fail here: closing a buffered stdin
                        # flushes it, and a worker killed between write() and flush()
                        # (for example by the OOM killer) leaves it broken. That must
                        # not shadow the caller's own exception with a bare EPIPE.
                        with contextlib.suppress(OSError):
                            stream.close()
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None

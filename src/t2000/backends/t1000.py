"""Reuse the installed T-1000 model/runtime in its own persistent process."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from .base import AudioChunk
from .f5 import F5Backend, sha256
from ..errors import T2000Error


MODEL_PINS = {
    "soar": {"directory": "dots.tts-soar", "revision": "2f9b3e18d70d670d4c701da2dc55ded5755815ce",
             "sha256": "2787e6d4fe0b27ac33d28072abcadef53802c841bff037f3ee1f6bb5e1d3a2ce"},
    "mf2": {"directory": "dots.tts-mf-2steps", "revision": "159b33d33de0f9610d9ea73725a0820d27261fd7",
            "sha256": "5022f36ee599f037efc2869e351bfe2bb627573e05b45d5b1c99437bc674351a"},
}
RUNTIME_REVISION = "b65ca7149fb43b8fb8d52916aa12738deda7b188"
MODEL_FILES = ("model.safetensors", "speaker_encoder.safetensors", "vocoder.safetensors", "config.json",
               "llm_config.json", "latent_stats.pt", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")
# Each profile loads its own runtime entry point; the rest of the package is shared.
# high-vram keeps the language model dense instead of quantizing it to NF4, which
# restores the upstream reference limit (dots.tts crops the speaker encoder at
# config.xvec_max_audio_seconds = 10).
PROFILE_SOURCES = {
    "low-vram": ("low_vram.py", "references.py", "eager.py", "vocoder.py"),
    "high-vram": ("high_vram.py", "references.py", "eager.py", "vocoder.py"),
}
PROFILE_REFERENCE_SECONDS = {"low-vram": 6, "high-vram": 10}
REFERENCE_BOUNDARY_POLICY = "neutral-terminal-punctuation-v1"


def condition_reference_text(text: str, boundary: str) -> str:
    """Add a neutral sentence end without changing spoken words or guessing names."""
    value = text.strip()
    if boundary == "raw":
        return value
    if boundary != "sentence":
        raise T2000Error("T-1000 reference boundary must be sentence or raw.")
    split = len(value)
    while split and value[split - 1] in '\"\'”’»›':
        split -= 1
    core, closing_quotes = value[:split], value[split:]
    if core.endswith((".", "!", "?", "…")):
        return value
    return core + "." + closing_quotes


class T1000Backend(F5Backend):
    """Share only F5's bounded process transport and shutdown, not its inference."""

    requires_reference_text = True

    def __init__(self, model_dir: Path, *, python_executable: Path, t1000_root: Path,
                 model="soar", device="cuda:0", num_threads=4, reference_boundary="sentence",
                 profile="low-vram", runtime_source=None):
        if model not in MODEL_PINS:
            raise T2000Error("T-1000 model must be soar or mf2.")
        if profile not in PROFILE_SOURCES:
            raise T2000Error("T-1000 profile must be low-vram or high-vram.")
        if not isinstance(device, str) or not device.startswith("cuda"):
            raise T2000Error("The reused T-1000 low-VRAM runtime requires CUDA.")
        if reference_boundary not in {"raw", "sentence"}:
            raise T2000Error("T-1000 reference boundary must be sentence or raw.")
        super().__init__(model_dir, python_executable=python_executable, device=device, num_threads=num_threads)
        self.t1000_root = Path(t1000_root).expanduser().resolve()
        self.model = model
        self.reference_boundary = reference_boundary
        self.profile = profile
        self.max_reference_seconds = PROFILE_REFERENCE_SECONDS[profile]
        # Vendored runtime, or the legacy external T-1000 checkout layout.
        self.runtime_source = (Path(runtime_source).expanduser().resolve()
                               if runtime_source else (self.t1000_root / "apps/api/src").resolve())
        self.bundle_dir = self.t1000_root / "data/models" / MODEL_PINS[model]["directory"]
        self._worker_script = self._project / "scripts/t1000_infer_worker.py"
        self._runtime_info = None

    def _identity(self):
        try:
            if not self.python_executable.is_file():
                raise ValueError("The existing T-1000 environment interpreter is missing")
            source = self.runtime_source / "t1000"
            paths = {"models/" + name: self.bundle_dir / name for name in MODEL_FILES}
            paths.update({"t1000/" + name: source / name for name in PROFILE_SOURCES[self.profile]})
            stamp = [(name, path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                     for name, path in sorted(paths.items())]
            if stamp != self._identity_stamp:
                hashes = {name: sha256(path) for name, path in sorted(paths.items())}
                if hashes["models/model.safetensors"] != MODEL_PINS[self.model]["sha256"]:
                    raise ValueError("T-1000 checkpoint does not match its pinned revision")
                config = json.loads((self.bundle_dir / "config.json").read_text())
                if self.model == "mf2" and config.get("sampling") != {
                    "solver": "scm", "ode_method": "euler", "num_steps": 2,
                    "guidance_scale": 0.0, "tau_mid": 1.3,
                }:
                    raise ValueError("MF2 trained sampling configuration does not match the pinned contract")
                self._identity_cache = {"files_sha256": hashes, "revision": MODEL_PINS[self.model]["revision"]}
                self._identity_stamp = stamp
            if self._runtime_info is None:
                # bitsandbytes is absent on high-vram, which keeps the language model dense
                # instead of quantizing it to NF4; a missing optional package is reported as
                # null rather than failing the whole probe.
                command = (
                    "import importlib.metadata as m,json\n"
                    "d=m.distribution('dots-tts')\n"
                    "def v(n):\n"
                    "    try: return m.version(n)\n"
                    "    except m.PackageNotFoundError: return None\n"
                    "print(json.dumps({'versions':{n:v(n) for n in "
                    "['torch','torchaudio','dots-tts','bitsandbytes','accelerate']},"
                    "'dots_direct_url':json.loads(d.read_text('direct_url.json') or '{}')}))\n"
                )
                result = subprocess.run([str(self.python_executable), "-c", command],
                                        capture_output=True, text=True, timeout=30, check=True)
                self._runtime_info = json.loads(result.stdout)
                if self._runtime_info["dots_direct_url"].get("vcs_info", {}).get("commit_id") != RUNTIME_REVISION:
                    raise ValueError("T-1000 dots-tts runtime revision differs from the validated installation")
            return {"models": self._identity_cache, "runtime": self._runtime_info,
                    "model": self.model, "python": str(self.python_executable),
                    "t1000_root": str(self.t1000_root), "device": self.device, "num_threads": self.num_threads,
                    "worker_sha256": sha256(self._worker_script), "adapter_sha256": sha256(Path(__file__)),
                    "profile": self.profile, "runtime_source": str(self.runtime_source),
                    "reference_trim": False, "precision": "bfloat16",
                    "llm": "nf4" if self.profile == "low-vram" else "bfloat16",
                    "reference_boundary": self.reference_boundary, "reference_boundary_policy": REFERENCE_BOUNDARY_POLICY,
                    "sampling": "SOAR10/CFG1.2" if self.model == "soar" else "trained MF2 defaults"}
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            raise T2000Error(f"Existing T-1000 models/runtime are unavailable: {exc}. "
                             "Select the installed T-1000 root and its existing virtual environment.") from exc

    def signature(self):
        return hashlib.sha256(json.dumps(self._identity(), sort_keys=True).encode()).hexdigest()

    def _exchange(self, request, timeout=600):
        try:
            return super()._exchange(request, timeout)
        except T2000Error as exc:
            raise T2000Error(str(exc).replace("F5", "T-1000")) from exc

    def load(self):
        with self._lock:
            signature = self.signature()
            if self._process is not None and self._process.poll() is None:
                if self._loaded_signature != signature:
                    raise T2000Error("T-1000 model files/settings changed after load. Restart before generating.")
                return
            self.close()
            self._stderr = tempfile.TemporaryFile()
            try:
                self._process = subprocess.Popen(
                    [str(self.python_executable), str(self._worker_script), "--t1000-root", str(self.t1000_root),
                     "--model", self.model, "--device", self.device, "--num-threads", str(self.num_threads),
                     "--profile", self.profile, "--runtime-source", str(self.runtime_source)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, text=True, bufsize=1,
                    cwd=self._project,
                )
                self._exchange({"command": "load"})
                self._loaded_signature = signature
            except OSError as exc:
                details = self._diagnostics()
                self.close()
                raise T2000Error(f"Could not start the T-1000 worker: {exc}"
                                 + (f" ({details})" if details else "") + ".") from exc
            except BaseException:
                self.close()
                raise

    def prepare(self, reference_path: Path, cache_dir: Path, *, reference_text=""):
        if not isinstance(reference_text, str) or not reference_text.strip() or len(reference_text) > 2000:
            raise T2000Error("T-1000 needs the exact reference transcript. Check the recorded words.")
        conditioning_text = condition_reference_text(reference_text, self.reference_boundary)
        reference_path = Path(reference_path).expanduser().resolve()
        try:
            info = sf.info(reference_path)
        except (OSError, RuntimeError) as exc:
            raise T2000Error(f"Could not read the T-1000 reference: {exc}") from exc
        if not 1 <= info.duration <= self.max_reference_seconds:
            raise T2000Error("Use a complete 1–6 second T-1000 low-VRAM reference with its matching words.")
        with self._lock:
            self.load()
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            value = {"schema": 2, "signature": self._loaded_signature, "reference_path": str(reference_path),
                     "reference_sha256": sha256(reference_path), "reference_text": conditioning_text,
                     "source_transcript": reference_text, "conditioning_text": conditioning_text,
                     "reference_boundary": self.reference_boundary, "reference_boundary_policy": REFERENCE_BOUNDARY_POLICY}
            destination = cache_dir / "t1000-reference.json"
            if destination.is_file():
                try:
                    if json.loads(destination.read_text()) == value:
                        return
                except (OSError, ValueError):
                    pass
            self._exchange({"command": "prepare", **value})
            fd, temporary = tempfile.mkstemp(prefix=".t1000-reference-", suffix=".json.tmp", dir=cache_dir)
            try:
                with os.fdopen(fd, "w") as output:
                    output.write(json.dumps(value, indent=2) + "\n")
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)

    def synthesize(self, text, cache_dir, *, speed, seed):
        if not isinstance(text, str) or not text.strip() or len(text) > 600:
            raise T2000Error("T-1000 needs a nonempty phrase of at most 600 characters.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise T2000Error("T-1000 seed must be an integer between 0 and 4294967295.")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed != 1:
            raise T2000Error("T-1000 preserves its native delivery; use speed 1.0 for this comparison.")
        with self._lock:
            self.load()
            try:
                profile = json.loads((Path(cache_dir) / "t1000-reference.json").read_text())
                if (not isinstance(profile, dict) or profile.get("schema") != 2
                        or profile.get("signature") != self._loaded_signature
                        or not isinstance(profile.get("reference_text"), str)
                        or not profile["reference_text"].strip()
                        or sha256(profile["reference_path"]) != profile["reference_sha256"]):
                    raise ValueError("T-1000 reference or settings changed")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise T2000Error(f"T-1000 reference preparation is unavailable ({exc}); prepare this voice again.") from exc
            with tempfile.TemporaryDirectory(prefix="t1000-output-", dir=cache_dir) as temporary:
                output_path = Path(temporary) / "speech.wav"
                response = self._exchange({"reference_path": profile["reference_path"],
                                           "reference_text": profile["reference_text"], "text": text,
                                           "seed": seed, "output_path": str(output_path.resolve())})
                if response.get("stop_reason") != "eos":
                    raise T2000Error("T-1000 reached its generation limit without completing the sentence.")
                info = sf.info(output_path)
                if info.samplerate != 48000 or info.channels != 1 or not 0 < info.duration <= 600:
                    raise T2000Error("T-1000 returned invalid audio dimensions.")
                samples, rate = sf.read(output_path, dtype="float32", always_2d=True)
                if (not samples.size or not np.isfinite(samples).all() or not np.any(samples)
                        or np.any(np.abs(samples) >= 32767 / 32768)):
                    raise T2000Error("T-1000 returned invalid, silent or clipped audio.")
                seconds = response.get("generation_seconds")
                if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) \
                        or seconds < 0:
                    raise T2000Error("T-1000 returned invalid generation timing.")
                self._last_metrics = {key: response.get(key) for key in (
                    "cuda_memory", "chunks", "raw_peak", "raw_clipped_fraction", "output_gain",
                    "first_pcm_seconds", "generation_seconds", "stop_reason", "provenance",
                )}
                self._last_metrics["reference_conditioning"] = {key: profile.get(key) for key in (
                    "source_transcript", "conditioning_text", "reference_boundary", "reference_boundary_policy",
                    "reference_sha256",
                )}
                return AudioChunk(np.ascontiguousarray(samples[:, 0]), rate, seconds, 0.0)

    def info(self):
        return {"backend": f"t1000-{self.model}", "loaded": self._process is not None
                and self._process.poll() is None and self._loaded_signature is not None,
                "model_dir": str(self.bundle_dir), "t1000_root": str(self.t1000_root),
                "tts_device": self.device, "converter_device": None, "sample_rate": 48000,
                "reference_mode": "whole audio and exact transcript", "requires_reference_text": True,
                "reference_boundary": self.reference_boundary, "reference_boundary_policy": REFERENCE_BOUNDARY_POLICY,
                "max_reference_seconds": self.max_reference_seconds, "signature": self._loaded_signature,
                "streaming": "complete adapter phrases", "last_worker": self._last_metrics}

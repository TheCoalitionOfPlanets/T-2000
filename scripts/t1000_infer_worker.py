#!/usr/bin/env python3
"""Isolated, offline reuse of the existing T-1000 environment and model files.

JSON lines: load/prepare/synthesize/info/close. No packages or weights are installed.
All diagnostics go to stderr; stdout is reserved for bounded protocol responses.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time


PINS = {
    "soar": ("dots.tts-soar", "2f9b3e18d70d670d4c701da2dc55ded5755815ce",
             "2787e6d4fe0b27ac33d28072abcadef53802c841bff037f3ee1f6bb5e1d3a2ce"),
    "mf2": ("dots.tts-mf-2steps", "159b33d33de0f9610d9ea73725a0820d27261fd7",
            "5022f36ee599f037efc2869e351bfe2bb627573e05b45d5b1c99437bc674351a"),
}
RUNTIME_REVISION = "b65ca7149fb43b8fb8d52916aa12738deda7b188"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_request(request, *, generation=True):
    if not isinstance(request, dict):
        raise ValueError("Send a JSON object.")
    result = dict(request)
    limits = {"reference_path": 4096, "reference_text": 2000}
    if generation:
        limits.update(text=600, output_path=4096)
    for key, limit in limits.items():
        value = result.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"Provide nonempty {key} of at most {limit} characters.")
        result[key] = value.strip()
    if not Path(result["reference_path"]).is_file():
        raise ValueError("Reference audio does not exist.")
    seed = result.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 4294967295.")
    result["seed"] = seed
    if generation:
        path = Path(result["output_path"])
        if path.suffix.lower() != ".wav" or path.exists():
            raise ValueError("Use a fresh output path ending in .wav.")
    return result


class DecodeObserver:
    def __init__(self, model):
        self.original = model._decode
        self.stop_reason = "not_started"

    def __call__(self, *args, **kwargs):
        state = kwargs["state"]
        self.stop_reason = "decoding"
        stream = self.original(*args, **kwargs)
        try:
            yield from stream
        except GeneratorExit:
            self.stop_reason = "closed"
            raise
        except BaseException:
            self.stop_reason = "error"
            raise
        else:
            self.stop_reason = "eos" if state.end_flag else "schedule_exhausted"
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()


class Runtime:
    def __init__(self, root, model, device, num_threads):
        self.root = Path(root).expanduser().resolve()
        self.model_name, self.device, self.num_threads = model, device, num_threads
        self.model_path = self.root / "data/models" / PINS[model][0]
        self.loaded = False
        self.provenance = None
        self.load_seconds = None

    def imports(self):
        sys.dont_write_bytecode = True  # Reuse the existing checkout without writing import caches into it.
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                          T1000_REFERENCE_TRIM="0", T1000_TTS_PROFILE="low-vram", T1000_TTS_OPTIMIZE="0",
                          T1000_TTS_VOCODER_BATCH_SIZE="4", T1000_TTS_EAGER_CACHE="1",
                          T1000_TTS_COMPILE_VOCODER="0", T1000_ROOT=str(self.root))
        source = self.root / "apps/api/src"
        if not (source / "t1000/low_vram.py").is_file():
            raise ValueError("The existing T-1000 source checkout is missing.")
        sys.path.insert(0, str(source))
        direct_url = json.loads(metadata.distribution("dots-tts").read_text("direct_url.json") or "{}")
        if direct_url.get("vcs_info", {}).get("commit_id") != RUNTIME_REVISION:
            raise ValueError("The installed dots-tts runtime revision does not match T-1000's validated pin.")
        import numpy as np
        import soundfile as sf
        import torch
        from dots_tts.utils.util import seed_everything
        from t1000.eager import configure_eager_cache
        from t1000.low_vram import load_low_vram_runtime
        from t1000.references import configure_prompt_audio

        self.np, self.sf, self.torch = np, sf, torch
        self.seed_everything = seed_everything
        self.load_runtime = load_low_vram_runtime
        self.configure_prompt = configure_prompt_audio
        self.configure_cache = configure_eager_cache
        return {name: metadata.version(name) for name in
                ("torch", "torchaudio", "dots-tts", "bitsandbytes", "accelerate", "numpy", "soundfile")}

    def load(self):
        if self.loaded:
            return
        started = time.perf_counter()
        versions = self.imports()
        torch = self.torch
        if not self.device.startswith("cuda") or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("T-1000 low-VRAM reuse requires CUDA with BF16 support.")
        torch.cuda.set_device(torch.device(self.device))
        torch.set_num_threads(self.num_threads)
        files = {path.name: sha256(path) for path in self.model_path.iterdir()
                 if path.is_file() and not path.name.startswith(".")}
        if files.get("model.safetensors") != PINS[self.model_name][2]:
            raise ValueError("Existing T-1000 model checkpoint does not match its pinned SHA-256.")
        config = json.loads((self.model_path / "config.json").read_text())
        if self.model_name == "mf2" and config.get("sampling") != {
            "solver": "scm", "ode_method": "euler", "num_steps": 2,
            "guidance_scale": 0.0, "tau_mid": 1.3,
        }:
            raise ValueError("MF2 trained sampling settings do not match its pinned contract.")
        torch.cuda.reset_peak_memory_stats(self.device)
        self.runtime = self.load_runtime(self.model_path, precision="bfloat16")
        self.configure_prompt(self.runtime)
        self.configure_cache(self.runtime, self.model_name == "soar")
        torch.cuda.synchronize(self.device)
        import dots_tts.runtime as installed_runtime

        self.provenance = {
            "model": self.model_name, "model_dir": str(self.model_path), "model_revision": PINS[self.model_name][1],
            "files_sha256": files, "dots_runtime_revision": RUNTIME_REVISION, "versions": versions,
            "runtime_source_sha256": sha256(installed_runtime.__file__),
            "t1000_source_sha256": {name: sha256(self.root / "apps/api/src/t1000" / name) for name in
                                     ("low_vram.py", "references.py", "eager.py", "vocoder.py")},
            "precision": "BF16 acoustic core; FP32 speaker encoder/vocoder; NF4 language-model blocks",
            "reference_processing": "Whole mono prompt, no trimming or gain changes; T-1000 HQ resampling",
            "sampling": {"num_steps": 10, "guidance_scale": 1.2, "ode_method": "euler"}
                        if self.model_name == "soar" else config["sampling"],
            "sample_rate": self.runtime.sample_rate, "device": str(self.runtime.device),
        }
        self.load_seconds = time.perf_counter() - started
        self.loaded = True

    def prepare(self, request):
        request = validate_request(request, generation=False)
        self.load()
        with self.sf.SoundFile(request["reference_path"]) as recording:
            if not 1 <= recording.frames / recording.samplerate <= 6 or not 1 <= recording.channels <= 8:
                raise ValueError("Use a complete 1–6 second reference with its exact transcript.")
            samples = recording.read(dtype="float32", always_2d=True)
        if not self.np.isfinite(samples).all() or not samples.size or not self.np.any(samples):
            raise ValueError("Reference audio is empty, silent or non-finite.")
        # Cache only CPU prompt preprocessing here; no original reference is modified.
        self.runtime._load_prompt_audio(request["reference_path"])
        return {"ok": True, "type": "prepared", "reference_sha256": sha256(request["reference_path"]),
                "reference_text": request["reference_text"]}

    def generate(self, request):
        request = validate_request(request)
        started = time.perf_counter()
        self.prepare(request)
        torch, np = self.torch, self.np
        options = {"text": request["text"], "prompt_audio_path": request["reference_path"],
                   "prompt_text": request["reference_text"], "language": "en", "normalize_text": False,
                   "profile_inference": False, "log_profile_calls": False}
        if self.model_name == "soar":
            options.update(num_steps=10, guidance_scale=1.2, ode_method="euler")
        prepared = self.runtime._prepare_inputs(
            text=options["text"], prompt_audio_path=options["prompt_audio_path"],
            prompt_text=options["prompt_text"], template_name=None, language="en", normalize_text=False,
        )
        original_prepare = self.runtime._prepare_inputs
        self.runtime._prepare_inputs = lambda **unused: prepared
        observer = DecodeObserver(self.runtime.model)
        self.runtime.model._decode = observer
        chunks, first_pcm = [], None
        self.seed_everything(request["seed"])
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        generation_started = time.perf_counter()
        try:
            with torch.inference_mode():
                stream = self.runtime.generate_stream(**options)
                try:
                    for chunk in stream:
                        pcm = chunk.detach().float().cpu().reshape(-1)
                        if not pcm.numel():
                            continue
                        if first_pcm is None:
                            first_pcm = time.perf_counter() - generation_started
                        if not torch.isfinite(pcm).all():
                            raise ValueError("T-1000 produced non-finite audio.")
                        chunks.append(pcm)
                        if sum(part.numel() for part in chunks) > self.runtime.sample_rate * 120:
                            raise ValueError("T-1000 exceeded the 120-second output limit.")
                finally:
                    stream.close()
            torch.cuda.synchronize(self.device)
        finally:
            self.runtime._prepare_inputs = original_prepare
            self.runtime.model._decode = observer.original
        generation_seconds = time.perf_counter() - generation_started
        if observer.stop_reason != "eos":
            raise ValueError(f"T-1000 did not complete the sentence: {observer.stop_reason}.")
        samples = torch.cat(chunks).numpy() if chunks else np.empty(0, dtype=np.float32)
        if not samples.size or not np.isfinite(samples).all() or not np.any(samples):
            raise ValueError("T-1000 produced empty, silent or non-finite audio.")
        raw_peak = float(np.max(np.abs(samples)))
        raw_clipped = float(np.mean(np.abs(samples) >= 1))
        output_gain = min(1.0, 0.98 / raw_peak)
        samples = np.ascontiguousarray(samples * output_gain, dtype=np.float32)
        path = Path(request["output_path"]).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".wav.part", dir=path.parent)
        os.close(fd)
        try:
            self.sf.write(temporary, samples, self.runtime.sample_rate, format="WAV", subtype="PCM_16")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {"ok": True, "type": "complete", "wav_path": str(path), "stop_reason": observer.stop_reason,
                "sample_rate": self.runtime.sample_rate, "generation_seconds": generation_seconds,
                "first_pcm_seconds": first_pcm, "audio_seconds": samples.size / self.runtime.sample_rate,
                "request_seconds": time.perf_counter() - started,
                "rtf": generation_seconds / (samples.size / self.runtime.sample_rate), "chunks": len(chunks),
                "raw_peak": raw_peak, "raw_clipped_fraction": raw_clipped, "output_gain": output_gain,
                "peak_before_clip": float(np.max(np.abs(samples))), "clipped_fraction": 0.0,
                "reference_sha256": sha256(request["reference_path"]), "seed": request["seed"],
                "cuda_memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                                "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)},
                "provenance": self.provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1000-root", type=Path, required=True)
    parser.add_argument("--model", choices=PINS, default="soar")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--check-imports", action="store_true", help="CPU-only imports; no model load")
    args = parser.parse_args()
    if not 1 <= args.num_threads <= 32:
        parser.error("Use between 1 and 32 CPU threads.")
    runtime = Runtime(args.t1000_root, args.model, args.device, args.num_threads)
    if args.check_imports:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        with contextlib.redirect_stdout(sys.stderr):
            versions = runtime.imports()
        print(json.dumps({"ok": True, "versions": versions}), flush=True)
        return 0
    while line := sys.stdin.readline(65537):
        request = {}
        try:
            if len(line) > 65536:
                raise ValueError("Worker request exceeds 64 KiB.")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Send a JSON object.")
            command = request.get("command", "synthesize")
            with contextlib.redirect_stdout(sys.stderr):
                if command == "load":
                    runtime.load()
                    response = {"ok": True, "type": "loaded", "load_seconds": runtime.load_seconds,
                                "provenance": runtime.provenance}
                elif command == "prepare":
                    response = runtime.prepare(request)
                elif command == "info":
                    response = {"ok": True, "type": "info", "loaded": runtime.loaded,
                                "model": runtime.model_name, "model_dir": str(runtime.model_path)}
                elif command == "close":
                    response = {"ok": True, "type": "closed"}
                elif command == "synthesize":
                    response = runtime.generate(request)
                else:
                    raise ValueError("Unknown worker command.")
        except Exception as exc:
            response = {"ok": False, "type": "error", "error": f"{type(exc).__name__}: {exc}"}
        if isinstance(request, dict) and "request_id" in request:
            response["request_id"] = request["request_id"]
        print(json.dumps(response, allow_nan=False), flush=True)
        if response.get("type") == "closed":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

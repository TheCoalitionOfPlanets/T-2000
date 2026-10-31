#!/usr/bin/env python3
"""Persistent JSON-lines F5-TTS worker for the isolated .venv-f5 environment.

Input: {reference_path, reference_text, text, output_path, seed?, speed?, steps?}.
Commands load/info/close are also supported. Stdout contains one JSON response
per request; imports and upstream diagnostics go to stderr. No models or ASR
resources are downloaded by this process.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time


SAMPLE_RATE = 24000
HOP_LENGTH = 256
MAX_TOTAL_SECONDS = 27
WORKER_VERSION = "f5-v1-local-worker-3-context-safe-joins"
MODEL_ARCHITECTURE = {
    "dim": 1024, "depth": 22, "heads": 16, "ff_mult": 2, "text_dim": 512,
    "text_mask_padding": True, "qk_norm": None, "conv_layers": 4,
    "pe_attn_head": None, "attn_backend": "torch", "attn_mask_enabled": False,
    "checkpoint_activations": False,
}


def validate_request(request: dict) -> dict:
    if not isinstance(request, dict):
        raise ValueError("Send a JSON object.")
    result = dict(request)
    for name, limit in (("reference_path", 4096), ("output_path", 4096),
                        ("reference_text", 2000), ("text", 600)):
        value = result.get(name)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"Provide nonempty {name} of at most {limit} characters.")
        result[name] = value.strip()
    if not Path(result["reference_path"]).is_file():
        raise ValueError("Reference WAV does not exist.")
    if Path(result["output_path"]).suffix.lower() != ".wav":
        raise ValueError("Output path must end in .wav.")
    if Path(result["output_path"]).exists():
        raise ValueError("Use a new output WAV path to preserve earlier results.")
    seed = result.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 4294967295.")
    speed = result.get("speed", 1.0)
    if (isinstance(speed, bool) or not isinstance(speed, (int, float))
            or not math.isfinite(speed) or not 0.5 <= speed <= 2.0):
        raise ValueError("Speed must be a finite number between 0.5 and 2.")
    steps = result.get("steps", 32)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 8 <= steps <= 64:
        raise ValueError("Use an integer between 8 and 64 F5 sampling steps.")
    result.update(seed=seed, speed=float(speed), steps=steps)
    preprocessing = result.get("reference_preprocessing", "upstream-edges")
    if preprocessing not in {"raw", "upstream-edges"}:
        raise ValueError("Use raw or upstream-edges reference preprocessing.")
    result["reference_preprocessing"] = preprocessing
    return result


def estimated_duration_frames(reference_frames: int, reference_text: str, text: str, speed: float) -> int:
    """Upstream duration heuristic: reference frames per UTF-8 text byte.

    Short text uses upstream's lower local speed. This predicts duration, not
    phoneme alignment or exact imitation of a reference's intonation contour.
    """
    reference_length = len(reference_text.encode("utf-8"))
    if reference_length == 0:
        raise ValueError("An exact reference transcript is required.")
    local_speed = speed * 0.3 if len(text.encode("utf-8")) < 10 else speed
    duration = reference_frames + int(reference_frames / reference_length * len(text.encode("utf-8")) / local_speed)
    if duration <= reference_frames or duration * HOP_LENGTH / SAMPLE_RATE > MAX_TOTAL_SECONDS:
        raise ValueError("This request exceeds the bounded F5 context. Use a shorter sentence or faster delivery.")
    return duration


def plan_text_batches(reference_frames: int, reference_text: str, text: str, speed: float) -> list[str]:
    """Bound model context using the reference pace without dropping words."""
    try:
        estimated_duration_frames(reference_frames, reference_text, text, speed)
        return [text]
    except ValueError:
        pass
    batches, current = [], ""
    for word in text.split():
        candidate = (current + " " + word).strip()
        try:
            estimated_duration_frames(reference_frames, reference_text, candidate, speed)
            current = candidate
        except ValueError:
            if not current:
                raise ValueError("A word exceeds this reference's F5 context. Check the reference transcript.") from None
            batches.append(current)
            estimated_duration_frames(reference_frames, reference_text, word, speed)
            current = word
    if current:
        batches.append(current)
    if not batches or len(batches) > 12:
        raise ValueError("Use shorter text or a clearer reference transcript for this bounded F5 request.")
    return batches


def reference_bounds(samples, rate):
    """Pinned upstream's -42 dBFS, 10 ms edge trim, without changing the WAV."""
    hop, threshold = max(1, round(rate * 0.01)), 10 ** (-42 / 20)
    start, end = 0, len(samples)
    while start + hop < end:
        frame = samples[start:start + hop].astype("float64")
        if math.sqrt(float((frame ** 2).mean())) >= threshold:
            break
        start += hop
    while end - hop > start:
        frame = samples[end - hop:end].astype("float64")
        if math.sqrt(float((frame ** 2).mean())) >= threshold:
            break
        end -= hop
    return start, end


def assemble_chunks(chunks, texts, rate=SAMPLE_RATE):
    """Retain complete speech at boundaries; only add missing punctuation silence.

    Overlapping independently generated speech can blend or lose consonants.
    This join leaves every generated frame intact, including natural pauses.
    """
    import numpy as np

    if not chunks or len(chunks) != len(texts):
        raise ValueError("Each generated chunk needs its original text.")

    def quiet_seconds(samples, trailing=False):
        if trailing:
            samples = samples[::-1]
        width = max(1, round(rate * 0.01))
        frames = 0
        for offset in range(0, min(samples.size, rate), width):
            block = samples[offset:offset + width].astype(np.float64)
            if not block.size or np.sqrt(np.mean(block ** 2)) >= 10 ** (-50 / 20):
                break
            frames += block.size
        return frames / rate

    output, inserted = [chunks[0]], 0
    for index, chunk in enumerate(chunks[1:], 1):
        previous = texts[index - 1].rstrip()
        desired = 0.16 if previous.endswith((".", "!", "?")) else 0.08 if previous.endswith((",", ";", ":")) else 0
        existing = quiet_seconds(chunks[index - 1], True) + quiet_seconds(chunk)
        missing = max(0, round((desired - existing) * rate))
        if missing:
            output.append(np.zeros(missing, dtype=np.float32))
            inserted += missing
        output.append(chunk)
    return np.concatenate(output), inserted / rate


class Runtime:
    def __init__(self, model_dir: Path, device: str, num_threads: int):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.device = device
        self.num_threads = num_threads
        self.loaded = False
        self.load_seconds = None
        self.provenance = None

    def imports(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        source = self.model_dir / "sources/F5-TTS/src"
        if not (source / "f5_tts/model/cfm.py").is_file():
            raise ValueError("F5 source is missing. Run scripts/setup_f5.py first.")
        sys.path.insert(0, str(source))
        import numpy as np
        import soundfile as sf
        import soxr
        import torch
        from f5_tts.model import CFM, DiT
        from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
        from safetensors.torch import load_file
        from vocos import Vocos

        self.np, self.sf, self.soxr, self.torch = np, sf, soxr, torch
        self.CFM, self.DiT, self.Vocos = CFM, DiT, Vocos
        self.convert_text, self.get_tokenizer, self.load_file = convert_char_to_pinyin, get_tokenizer, load_file
        return {name: metadata.version(name) for name in
                ("torch", "torchaudio", "numpy", "vocos", "x-transformers", "torchdiffeq", "safetensors")}

    def load(self):
        if self.loaded:
            return
        started = time.perf_counter()
        from setup_f5 import check_bundle

        manifest = check_bundle(self.model_dir)
        versions = self.imports()
        torch = self.torch
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("Use cpu or cuda:N for the F5 worker device.")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in the isolated F5 environment.")
        torch.set_num_threads(self.num_threads)
        vocabulary, size = self.get_tokenizer(str(self.model_dir / "vocab.txt"), "custom")
        self.model = self.CFM(
            transformer=self.DiT(**MODEL_ARCHITECTURE, text_num_embeds=size, mel_dim=100),
            mel_spec_kwargs={"n_fft": 1024, "hop_length": HOP_LENGTH, "win_length": 1024,
                             "n_mel_channels": 100, "target_sample_rate": SAMPLE_RATE, "mel_spec_type": "vocos"},
            odeint_kwargs={"method": "euler"}, vocab_char_map=vocabulary,
        )
        # Deserialize on CPU to avoid a second full checkpoint allocation in
        # VRAM during loading. EMA key normalization follows pinned upstream.
        weights = self.load_file(str(self.model_dir / "model.safetensors"), device="cpu")
        weights = {key.replace("ema_model.", ""): value for key, value in weights.items()
                   if key not in {"initted", "step"}}
        for key in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
            weights.pop(key, None)
        self.model.load_state_dict(weights, strict=True)
        del weights
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = self.model.to(device=self.device, dtype=dtype).eval()
        self.vocoder = self.Vocos.from_hparams(str(self.model_dir / "vocos/config.yaml"))
        weights = torch.load(self.model_dir / "vocos/pytorch_model.bin", map_location="cpu", weights_only=True)
        self.vocoder.load_state_dict(weights, strict=True)
        del weights
        self.vocoder = self.vocoder.to(self.device).eval()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.provenance = {
            "worker": WORKER_VERSION, "model": manifest["model"], "source_revision": manifest["source_revision"],
            "model_revision": manifest["model_revision"], "vocoder_revision": manifest["vocoder_revision"],
            "model_sha256": manifest["files"]["model.safetensors"], "license": manifest["license"],
            "device": self.device, "dtype": str(dtype), "versions": versions,
            "reference_conditioning": "Direct audio+text context; reference-based duration heuristic",
        }
        self.load_seconds = time.perf_counter() - started
        self.loaded = True

    def generate(self, request):
        request = validate_request(request)
        started = time.perf_counter()
        self.load()
        torch, np = self.torch, self.np
        preparation_started = time.perf_counter()
        reference_path = Path(request["reference_path"]).expanduser().resolve()
        with self.sf.SoundFile(reference_path) as recording:
            if not 1 <= recording.frames / recording.samplerate <= 12:
                raise ValueError("F5 requires a complete 1–12 second reference and its matching transcript.")
            reference = recording.read(dtype="float32", always_2d=True)
            rate = recording.samplerate
        if not np.isfinite(reference).all():
            raise ValueError("Reference audio contains invalid samples.")
        reference = reference.mean(axis=1, dtype=np.float64).astype(np.float32)
        original_seconds = len(reference) / rate
        start, end = (0, len(reference))
        if request["reference_preprocessing"] == "upstream-edges":
            start, end = reference_bounds(reference, rate)
            if (end - start) / rate < 0.5:
                raise ValueError("Too little audible reference speech remains after edge trimming.")
            reference = np.concatenate([reference[start:end], np.zeros(round(rate * 0.05), dtype=np.float32)])
        conditioning_seconds = len(reference) / rate
        rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
        if rms < 1e-5:
            raise ValueError("Reference audio is silent or too quiet.")
        if rate != SAMPLE_RATE:
            reference = self.soxr.resample(reference, rate, SAMPLE_RATE, quality="HQ")
        if rms < 0.1:
            reference = reference * (0.1 / rms)
        audio = torch.from_numpy(np.ascontiguousarray(reference, dtype=np.float32)).unsqueeze(0).to(self.device)
        reference_text = request["reference_text"]
        reference_text += " " if reference_text.endswith((".", "!", "?")) else ". "
        reference_frames = audio.shape[-1] // HOP_LENGTH
        batches = plan_text_batches(reference_frames, reference_text, request["text"], request["speed"])
        durations = [estimated_duration_frames(reference_frames, reference_text, batch, request["speed"])
                     for batch in batches]
        torch.manual_seed(request["seed"])
        np.random.seed(request["seed"])
        random.seed(request["seed"])
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        preparation_seconds = time.perf_counter() - preparation_started
        generation_started = time.perf_counter()
        chunks = []
        with torch.inference_mode():
            for index, (batch, duration) in enumerate(zip(batches, durations)):
                text = self.convert_text([reference_text + batch])
                generated, trajectory = self.model.sample(
                    cond=audio, text=text, duration=duration, steps=request["steps"],
                    cfg_strength=2.0, sway_sampling_coef=-1.0, seed=(request["seed"] + index) % 2**32,
                )
                del trajectory
                generated = generated[:, reference_frames:, :].to(torch.float32).permute(0, 2, 1)
                output = self.vocoder.decode(generated)
                if rms < 0.1:
                    output = output * (rms / 0.1)
                chunk = output.squeeze().detach().cpu().numpy().astype(np.float32)
                if chunk.ndim != 1 or not chunk.size or not np.isfinite(chunk).all():
                    raise ValueError("F5 produced invalid or empty audio.")
                chunks.append(chunk)
        samples, inserted_pause_seconds = assemble_chunks(chunks, batches)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_started
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise ValueError("F5 produced invalid or empty audio.")
        peak = float(np.max(np.abs(samples)))
        if peak < 1e-7:
            raise ValueError("F5 produced silent audio.")
        raw_peak = peak
        raw_clipped_fraction = float(np.mean(np.abs(samples) >= 1))
        output_gain = min(1.0, 0.98 / peak)
        samples = samples * output_gain
        peak = float(np.max(np.abs(samples)))
        output_path = Path(request["output_path"]).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{output_path.stem}-", suffix=".wav.part", dir=output_path.parent)
        os.close(fd)
        try:
            self.sf.write(temporary, np.clip(samples, -1, 1), SAMPLE_RATE, format="WAV", subtype="PCM_16")
            os.replace(temporary, output_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        memory = {}
        if self.device.startswith("cuda"):
            memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}
        return {"ok": True, "type": "complete", "wav_path": str(output_path), "sample_rate": SAMPLE_RATE,
                "audio_seconds": samples.size / SAMPLE_RATE, "generation_seconds": generation_seconds,
                "rtf": generation_seconds / (samples.size / SAMPLE_RATE), "first_pcm_seconds": generation_seconds,
                "load_seconds": self.load_seconds, "preparation_seconds": preparation_seconds,
                "request_seconds": time.perf_counter() - started, "peak_before_clip": peak,
                "clipped_fraction": float(np.mean(np.abs(samples) >= 1)), "cuda_memory": memory,
                "raw_peak": raw_peak, "raw_clipped_fraction": raw_clipped_fraction, "output_gain": output_gain,
                "provenance": self.provenance, "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                "seed": request["seed"], "speed": request["speed"], "steps": request["steps"],
                "reference_preprocessing": request["reference_preprocessing"],
                "reference_original_seconds": original_seconds, "reference_conditioning_seconds": conditioning_seconds,
                "reference_trim_start_seconds": start / rate, "reference_trim_end_seconds": end / rate,
                "chunks": len(batches), "chunk_texts": batches,
                "join_policy": "Preserve complete speech; add only missing punctuation silence",
                "inserted_pause_seconds": inserted_pause_seconds,
                "estimated_total_context_seconds": [duration * HOP_LENGTH / SAMPLE_RATE for duration in durations]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/f5"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--check-imports", action="store_true", help="CPU import preflight; no weights or inference")
    args = parser.parse_args()
    if not 1 <= args.num_threads <= 32:
        parser.error("Use between 1 and 32 CPU threads.")
    runtime = Runtime(args.model_dir, args.device, args.num_threads)
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
                elif command == "info":
                    response = {"ok": True, "type": "info", "loaded": runtime.loaded,
                                "model_dir": str(runtime.model_dir), "provenance": runtime.provenance}
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

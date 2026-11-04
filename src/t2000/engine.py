import json
import importlib.util
from dataclasses import replace
import gc
import logging
import math
import threading
import time
import uuid
from collections.abc import Callable

import numpy as np
import soundfile as sf

from .backends.base import VoiceBackend
from .config import Settings
from .errors import T2000Error
from .profiles import ProfileStore
from .text import split_phrases

logger = logging.getLogger(__name__)

# Longest silence the studio will insert between sentences, in seconds.
MAX_SENTENCE_GAP = 2.0


class Cancelled(T2000Error):
    pass


class Engine:
    def __init__(self, settings: Settings, store: ProfileStore, backend: VoiceBackend | None = None):
        self.settings = settings
        self.store = store
        self._backend = backend
        self._lock = threading.Lock()
        self.loaded = False
        self._reference_delivery = None

    @property
    def backend(self) -> VoiceBackend:
        if self._backend is None:
            if self.settings.backend in {"t1000-soar", "t1000-mf2"}:
                from .backends.t1000 import T1000Backend

                self._backend = T1000Backend(
                    self.settings.model_dir, python_executable=self.settings.t1000_python,
                    t1000_root=self.settings.t1000_root,
                    model="soar" if self.settings.backend == "t1000-soar" else "mf2",
                    device=self.settings.tts_device, num_threads=self.settings.num_threads,
                    profile=self.settings.t1000_profile,
                    runtime_source=self.settings.t1000_runtime_source,
                )
                return self._backend
            if self.settings.backend == "f5-tts-v1":
                from .backends.f5 import F5Backend

                self._backend = F5Backend(
                    self.settings.model_dir, python_executable=self.settings.f5_python,
                    device=self.settings.tts_device, num_threads=self.settings.num_threads,
                    steps=self.settings.f5_steps,
                )
                return self._backend
            from .backends.melo_openvoice import MeloOpenVoiceBackend

            self._backend = MeloOpenVoiceBackend(
                self.settings.model_dir,
                tts_device=self.settings.tts_device,
                converter_device=self.settings.converter_device,
                speaker=self.settings.speaker,
                language=self.settings.language,
                num_threads=self.settings.num_threads,
                reference_mode=self.settings.reference_mode,
                tau=self.settings.conversion_tau,
            )
        return self._backend

    def status(self) -> dict:
        return {
            "backend": self.settings.backend,
            "loaded": self.loaded,
            "busy": self._lock.locked(),
            "devices": {"tts": self.settings.tts_device,
                        "converter": self.settings.converter_device if self.settings.backend == "melo-openvoice-v2"
                        else None},
            "speaker": self.settings.speaker if self.settings.backend == "melo-openvoice-v2" else "reference",
            "language": self.settings.language,
            "reference_mode": self.settings.reference_mode,
            "conversion_tau": self.settings.conversion_tau,
            "max_reference_seconds": self.max_reference_seconds,
            "available_backends": self.available_backends(),
            "delivery_planner": {"enabled": True, "type": "deterministic", "version": "context-v1"},
            "speed_control": {"enabled": not self.settings.backend.startswith("t1000-"),
                              "default": 1.0, "method": "native_reference" if
                              self.settings.backend.startswith("t1000-") else "model_pace"},
            "profiles_count": len(self.store.list()),
            "transport": "phrase PCM over WebSocket",
            "stt": {
                "enabled_by_default": True, "device": "cpu",
                "available": importlib.util.find_spec("sherpa_onnx") is not None and all(
                    (self.settings.stt_model_dir / filename).is_file()
                    for filename in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
                ),
            },
        }

    @property
    def max_reference_seconds(self) -> int:
        if self.settings.backend.startswith("t1000-"):
            from .backends.t1000 import PROFILE_REFERENCE_SECONDS

            return PROFILE_REFERENCE_SECONDS[self.settings.t1000_profile]
        return 12 if self.settings.backend == "f5-tts-v1" else 30

    def available_backends(self) -> list[dict]:
        root = self.settings.model_dir
        t1000 = self.settings.t1000_root
        return [
            {"id": "t1000-soar", "label": "T-1000 SOAR · Reference quality",
             "available": self.settings.tts_device.startswith("cuda") and self.settings.t1000_python.is_file()
             and (t1000 / "data/models/dots.tts-soar/model.safetensors").is_file()},
            {"id": "t1000-mf2", "label": "T-1000 MF2 · Faster reference",
             "available": self.settings.tts_device.startswith("cuda") and self.settings.t1000_python.is_file()
             and (t1000 / "data/models/dots.tts-mf-2steps/model.safetensors").is_file()},
            {"id": "f5-tts-v1", "label": "F5 · Direct voice",
             "available": self.settings.f5_python.is_file() and (root / "f5/manifest.json").is_file()},
            {"id": "melo-openvoice-v2", "label": "Melo/OpenVoice · Fast",
             "available": (root / "manifest.json").is_file()},
        ]

    def switch_backend(self, backend_id: str) -> dict:
        available = {item["id"]: item["available"] for item in self.available_backends()}
        if backend_id not in available or not available[backend_id]:
            raise T2000Error("That voice engine is not installed on this system.")
        if not self._lock.acquire(blocking=False):
            raise T2000Error("Wait for the current speech request before changing engines.")
        try:
            if backend_id != self.settings.backend:
                previous = self._backend
                self.close()
                self._backend = None
                del previous
                # Melo keeps an upstream BERT module-global reference. Release it
                # before another complete synthesis stack can occupy the GPU.
                import sys

                bert = sys.modules.get("melo.text.english_bert")
                if bert is not None:
                    bert.model = None
                gc.collect()
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.settings = replace(self.settings, backend=backend_id)
            result = self.status()
            result["busy"] = False  # The switch itself owns the idle model lock.
            return result
        finally:
            self._lock.release()

    def close(self) -> None:
        if self._backend is not None:
            close = getattr(self._backend, "close", None)
            if close is not None:
                close()
        self.loaded = False

    def _acquire(self, cancel: threading.Event) -> None:
        while not self._lock.acquire(timeout=0.1):
            self._check(cancel)
        if cancel.is_set():
            self._lock.release()
            raise Cancelled("Generation cancelled.")

    @staticmethod
    def _check(cancel: threading.Event) -> None:
        if cancel.is_set():
            raise Cancelled("Generation cancelled.")

    def _prepare(self, profile_id, cancel, emit):
        profile = self.store.get(profile_id)
        if getattr(self.backend, "requires_reference_text", False) and not profile["transcript"]:
            from .stt import transcribe_file

            emit({"type": "transcribing"})
            words = transcribe_file(
                self.store.reference_path(profile_id), self.settings.stt_model_dir,
                num_threads=self.settings.num_threads, max_seconds=30,
            )
            if not words.strip():
                raise T2000Error("No reference words were recognized. Type the exact recording transcript.")
            profile = self.store.update_transcript(profile_id, words, only_if_empty=True)
        self._check(cancel)
        emit({"type": "loading"})
        started = time.perf_counter()
        self.backend.load()
        load_seconds = time.perf_counter() - started
        self.loaded = True
        self._check(cancel)
        emit({"type": "preparing"})
        started = time.perf_counter()
        profile = self.store.get(profile_id)
        signature = self.backend.signature()
        cache_dir = self.store.conditioning_dir(profile_id, signature, transcript=profile["transcript"])
        self.backend.prepare(self.store.reference_path(profile_id), cache_dir, reference_text=profile["transcript"])
        self._reference_delivery = None
        if self.settings.backend != "melo-openvoice-v2":
            from .delivery import reference_delivery_profile

            try:
                self._reference_delivery = reference_delivery_profile(
                    self.store.reference_path(profile_id), profile["transcript"],
                    self.settings.model_dir, cache_dir,
                )
            except T2000Error as exc:
                # Optional diagnostics never replace the original reference or
                # prevent the neural backend from using its complete context.
                self._reference_delivery = {"available": False, "reason": str(exc)}
        prepare_seconds = time.perf_counter() - started
        self._check(cancel)
        return cache_dir, signature, load_seconds, prepare_seconds

    def prepare_profile(self, profile_id: str) -> dict:
        cancel = threading.Event()
        self._acquire(cancel)
        try:
            _, signature, load_seconds, prepare_seconds = self._prepare(profile_id, cancel, lambda _: None)
            return {"profile_id": profile_id, "signature": signature, "load_seconds": load_seconds,
                    "prepare_seconds": prepare_seconds, "ready": True,
                    "delivery_profile": self._reference_delivery}
        finally:
            self._lock.release()

    def generate(
        self, profile_id: str, text: str, *, speed: float = 1.0, seed: int = 42,
        sentence_gap: float = 0.0,
        emit: Callable[[dict], None] = lambda _: None, cancel: threading.Event | None = None,
    ) -> dict:
        """One serialized request. Emits metadata then little-endian float32 PCM.

        Cancellation is cooperative between model calls and transport packets.
        A cancelled/error output is never published as a complete WAV.
        """
        cancel = cancel if cancel is not None else threading.Event()
        if not isinstance(text, str) or len(text) > self.settings.max_text_chars:
            raise T2000Error(f"Text must contain at most {self.settings.max_text_chars} characters.")
        if not math.isfinite(speed) or not 0.7 <= speed <= 1.4:
            raise T2000Error("Speed must be between 0.7 and 1.4.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise T2000Error("Seed must be between 0 and 4294967295.")
        if (isinstance(sentence_gap, bool) or not isinstance(sentence_gap, (int, float))
                or not math.isfinite(sentence_gap) or not 0 <= sentence_gap <= MAX_SENTENCE_GAP):
            raise T2000Error(f"Sentence pause must be between 0 and {MAX_SENTENCE_GAP} seconds.")
        if not text.strip():
            raise T2000Error("Enter text to speak.")
        phrases = (split_phrases(text, self.settings.phrase_chars)
                   if self.settings.backend == "melo-openvoice-v2" else None)
        self.store.get(profile_id)
        submitted = time.perf_counter()
        emit({"type": "queued"})
        self._acquire(cancel)
        queue_seconds = time.perf_counter() - submitted
        job_id = uuid.uuid4().hex
        output_dir = self.settings.data_dir / "output"
        temporary = output_dir / f".{job_id}.wav.part"
        published = output_dir / f"{job_id}.wav"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_dir, signature, load_seconds, prepare_seconds = self._prepare(profile_id, cancel, emit)
            started = time.perf_counter()
            delivery_plan = None
            if self.settings.backend != "melo-openvoice-v2":
                from .delivery import plan_delivery

                profile = self.store.get(profile_id)
                delivery_plan = plan_delivery(
                    text, max_chars=600 if self.settings.backend == "f5-tts-v1" else 150,
                    reference_seconds=profile["duration_seconds"], reference_text=profile["transcript"],
                    speed=speed, max_context_seconds=27 if self.settings.backend == "f5-tts-v1" else None,
                )
                phrases = [segment["text"] for segment in delivery_plan["segments"]]
            first_pcm = None
            previous_pcm = None
            gaps = []
            frames = 0
            sample_rate = None
            sequence = 0
            base_seconds = conversion_seconds = 0.0
            peak = 0.0
            clipped_samples = 0
            silence_frames = 0
            audio_file = None
            try:
                for index, phrase in enumerate(phrases):
                    self._check(cancel)
                    emit({"type": "generating", "phrase": index + 1, "phrases": len(phrases)})
                    chunk = self.backend.synthesize(phrase, cache_dir, speed=speed, seed=(seed + index) % 2**32)
                    self._check(cancel)
                    pcm = np.asarray(chunk.samples, dtype=np.float32)
                    rate = int(chunk.sample_rate)
                    if pcm.ndim != 1 or not pcm.size or not np.isfinite(pcm).all():
                        raise T2000Error("The model produced invalid or empty audio.")
                    if not 8000 <= rate <= 96000 or (sample_rate is not None and sample_rate != rate):
                        raise T2000Error("The model changed its output sample rate.")
                    if (frames + pcm.size) / rate > self.settings.max_output_seconds:
                        raise T2000Error("Generated audio exceeded the 120-second limit. Use shorter text.")
                    chunk_peak = float(np.max(np.abs(pcm)))
                    if chunk_peak <= 1e-7:
                        raise T2000Error("The model produced silent audio.")
                    peak = max(peak, chunk_peak)
                    clipped_samples += int(np.count_nonzero(np.abs(pcm) >= 1.0))
                    if index and sentence_gap:
                        pause = np.zeros(round(sentence_gap * rate), dtype=np.float32)
                        if pause.size:
                            if (frames + pause.size + pcm.size) / rate > self.settings.max_output_seconds:
                                raise T2000Error("Generated audio exceeded the 120-second limit. "
                                                 "Use shorter text or a shorter sentence pause.")
                            if audio_file is None:
                                audio_file = sf.SoundFile(temporary, mode="w", samplerate=rate, channels=1,
                                                          format="WAV", subtype="PCM_16")
                            audio_file.write(pause)
                            frames += pause.size
                            silence_frames += pause.size
                            for offset in range(0, pause.size, max(1, rate // 10)):
                                self._check(cancel)
                                packet = pause[offset:offset + rate // 10].astype("<f4", copy=False)
                                emit({"type": "audio", "sample_rate": rate, "frames": packet.size,
                                      "sequence": sequence, "pcm": packet.tobytes()})
                                sequence += 1
                    now = time.perf_counter()
                    if first_pcm is None:
                        first_pcm = now - started
                    if previous_pcm is not None:
                        gaps.append(now - previous_pcm)
                    previous_pcm = now
                    sample_rate = rate
                    base_seconds += chunk.base_seconds
                    conversion_seconds += chunk.conversion_seconds
                    if audio_file is None:
                        audio_file = sf.SoundFile(temporary, mode="w", samplerate=rate, channels=1,
                                                  format="WAV", subtype="PCM_16")
                    # Export and playback share the same bounded samples.
                    pcm = np.clip(pcm, -1.0, 1.0)
                    audio_file.write(pcm)
                    frames += pcm.size
                    # Bound each transport packet to 100 ms; no claim of model chunk streaming.
                    for offset in range(0, pcm.size, max(1, rate // 10)):
                        self._check(cancel)
                        packet = pcm[offset:offset + rate // 10].astype("<f4", copy=False)
                        emit({"type": "audio", "sample_rate": rate, "frames": packet.size,
                              "sequence": sequence, "pcm": packet.tobytes()})
                        sequence += 1
                # End synthesis time when final PCM was produced, excluding final packet transport/export.
                generation_seconds = previous_pcm - started
            finally:
                if audio_file is not None:
                    audio_file.close()
            self._check(cancel)
            audio_seconds = frames / sample_rate
            spoken_seconds = (frames - silence_frames) / sample_rate
            metrics = {
                "queue_seconds": queue_seconds, "load_seconds": load_seconds,
                "prepare_seconds": prepare_seconds, "generation_seconds": generation_seconds,
                "first_pcm_seconds": first_pcm, "audio_seconds": audio_seconds,
                "rtf": generation_seconds / spoken_seconds, "base_seconds": base_seconds,
                "spoken_seconds": spoken_seconds, "sentence_gap_seconds": float(sentence_gap),
                "inserted_silence_seconds": silence_frames / sample_rate,
                "conversion_seconds": conversion_seconds, "phrases": len(phrases),
                "max_phrase_gap_seconds": max(gaps, default=0.0), "sample_rate": sample_rate,
                "peak_before_clip": peak, "clipped_fraction": clipped_samples / frames,
                "request_seconds": time.perf_counter() - submitted,
                "planning_seconds": delivery_plan["planning_seconds"] if delivery_plan else 0.0,
            }
            report = {"job_id": job_id, "profile_id": profile_id, "text": text, "seed": seed,
                      "speed": speed, "sentence_gap": float(sentence_gap), "backend_signature": signature, "backend": self.backend.info(),
                      "metrics": metrics, "complete": True,
                      "delivery_plan": delivery_plan, "reference_delivery": self._reference_delivery}
            report_tmp = output_dir / f".{job_id}.json.part"
            report_tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            self._check(cancel)
            report_path = output_dir / f"{job_id}.json"
            report_tmp.replace(report_path)
            try:
                self._check(cancel)
                # The WAV is the public artifact: promote it only after its report is durable.
                temporary.replace(published)
            except BaseException:
                try:
                    report_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove unpublished report %s", report_path, exc_info=True)
                raise
            return {"type": "complete", "job_id": job_id, "audio_url": f"/api/output/{job_id}.wav",
                    "metrics": metrics}
        finally:
            try:
                for unfinished in (temporary, output_dir / f".{job_id}.json.part"):
                    try:
                        unfinished.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Could not remove unfinished artifact %s", unfinished, exc_info=True)
            finally:
                self._lock.release()

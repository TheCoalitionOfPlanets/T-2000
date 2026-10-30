import json
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


class Cancelled(T2000Error):
    pass


class Engine:
    def __init__(self, settings: Settings, store: ProfileStore, backend: VoiceBackend | None = None):
        self.settings = settings
        self.store = store
        self._backend = backend
        self._lock = threading.Lock()
        self.loaded = False

    @property
    def backend(self) -> VoiceBackend:
        if self._backend is None:
            from .backends.melo_openvoice import MeloOpenVoiceBackend

            self._backend = MeloOpenVoiceBackend(
                self.settings.model_dir,
                tts_device=self.settings.tts_device,
                converter_device=self.settings.converter_device,
                speaker=self.settings.speaker,
                language=self.settings.language,
                num_threads=self.settings.num_threads,
            )
        return self._backend

    def status(self) -> dict:
        return {
            "backend": "melo-openvoice-v2",
            "loaded": self.loaded,
            "busy": self._lock.locked(),
            "devices": {"tts": self.settings.tts_device, "converter": self.settings.converter_device},
            "speaker": self.settings.speaker,
            "language": self.settings.language,
            "profiles_count": len(self.store.list()),
            "transport": "phrase PCM over WebSocket",
        }

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
        self.store.get(profile_id)
        self._check(cancel)
        emit({"type": "loading"})
        started = time.perf_counter()
        self.backend.load()
        load_seconds = time.perf_counter() - started
        self.loaded = True
        self._check(cancel)
        emit({"type": "preparing"})
        started = time.perf_counter()
        signature = self.backend.signature()
        cache_dir = self.store.conditioning_dir(profile_id, signature)
        self.backend.prepare(self.store.reference_path(profile_id), cache_dir)
        prepare_seconds = time.perf_counter() - started
        self._check(cancel)
        return cache_dir, signature, load_seconds, prepare_seconds

    def prepare_profile(self, profile_id: str) -> dict:
        cancel = threading.Event()
        self._acquire(cancel)
        try:
            _, signature, load_seconds, prepare_seconds = self._prepare(profile_id, cancel, lambda _: None)
            return {"profile_id": profile_id, "signature": signature, "load_seconds": load_seconds,
                    "prepare_seconds": prepare_seconds, "ready": True}
        finally:
            self._lock.release()

    def generate(
        self, profile_id: str, text: str, *, speed: float = 1.0, seed: int = 42,
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
        phrases = split_phrases(text, self.settings.phrase_chars)
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
            first_pcm = None
            previous_pcm = None
            gaps = []
            frames = 0
            sample_rate = None
            sequence = 0
            base_seconds = conversion_seconds = 0.0
            peak = 0.0
            clipped_samples = 0
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
            metrics = {
                "queue_seconds": queue_seconds, "load_seconds": load_seconds,
                "prepare_seconds": prepare_seconds, "generation_seconds": generation_seconds,
                "first_pcm_seconds": first_pcm, "audio_seconds": audio_seconds,
                "rtf": generation_seconds / audio_seconds, "base_seconds": base_seconds,
                "conversion_seconds": conversion_seconds, "phrases": len(phrases),
                "max_phrase_gap_seconds": max(gaps, default=0.0), "sample_rate": sample_rate,
                "peak_before_clip": peak, "clipped_fraction": clipped_samples / frames,
                "request_seconds": time.perf_counter() - submitted,
            }
            report = {"job_id": job_id, "profile_id": profile_id, "text": text, "seed": seed,
                      "speed": speed, "backend_signature": signature, "backend": self.backend.info(),
                      "metrics": metrics, "complete": True}
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

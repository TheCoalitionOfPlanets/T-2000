"""Bounded reference decoding and lightweight, descriptive audio diagnostics."""

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .errors import T2000Error


PREPROCESSING_VERSION = "mono-float32-native-v1"


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def load_audio(path: Path, max_seconds: float = 30, min_seconds: float = 1) -> Audio:
    """Decode a complete, bounded recording without resampling or changing gain."""
    if not (np.isfinite(max_seconds) and np.isfinite(min_seconds)
            and 0 <= min_seconds <= max_seconds and max_seconds > 0):
        raise ValueError("Audio duration bounds must be finite and ordered.")
    try:
        with sf.SoundFile(str(path)) as source:
            if not 8000 <= source.samplerate <= 192000:
                raise T2000Error("Use a recording with a sample rate between 8 kHz and 192 kHz.")
            if not 1 <= source.channels <= 8:
                raise T2000Error("Use a recording with one to eight audio channels.")
            duration = source.frames / source.samplerate
            if source.frames == 0 or duration < min_seconds:
                raise T2000Error(f"Use a reference recording of at least {min_seconds:g} seconds.")
            if duration > max_seconds:
                raise T2000Error(f"Use a complete reference recording of {max_seconds:g} seconds or less.")
            # Bound decoding even if a malformed file reports an incorrect frame count.
            limit = int(max_seconds * source.samplerate)
            samples = source.read(limit + 1, dtype="float32", always_2d=True)
            rate = source.samplerate
            if samples.shape[0] > limit:
                raise T2000Error(f"Use a complete reference recording of {max_seconds:g} seconds or less.")
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, T2000Error):
            raise
        raise T2000Error(f"Could not decode the reference audio: {exc}") from exc
    if not np.isfinite(samples).all():
        raise T2000Error("The reference recording contains invalid audio samples.")
    mono = samples.mean(axis=1, dtype=np.float64).astype(np.float32)
    if len(mono) < int(min_seconds * rate) or mono.size == 0:
        raise T2000Error(f"Use a reference recording of at least {min_seconds:g} seconds.")
    if float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) < 1e-5:
        raise T2000Error("The reference is silent or too quiet. Record a clear spoken sentence.")
    return Audio(np.ascontiguousarray(mono), int(rate))


def _waveform(samples: np.ndarray, rate: int) -> dict:
    # Min/max pairs retain peaks that point sampling can miss.
    boundaries = np.linspace(0, len(samples), min(300, len(samples)) + 1, dtype=int)
    indices = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        part = samples[start:end]
        indices.extend(sorted({start + int(np.argmin(part)), start + int(np.argmax(part))}))
    return {"time_seconds": [i / rate for i in indices],
            "amplitude": [float(samples[i]) for i in indices]}


def _pitch(samples: np.ndarray, rate: int) -> dict:
    target_rate = 8000
    divisor = gcd(rate, target_rate)
    values = signal.resample_poly(samples.astype(np.float64), target_rate // divisor, rate // divisor)
    width, hop = 320, 160  # 40 ms windows, 20 ms step.
    starts = np.arange(0, max(1, len(values) - width + 1), hop)
    if len(starts) > 600:
        starts = starts[np.linspace(0, len(starts) - 1, 600, dtype=int)]
    result = {"method": "Approximate autocorrelation F0; diagnostic only, not speaker identity or emotion.",
              "time_seconds": [], "f0_hz": [], "voiced": []}
    gate = max(1e-4, float(np.sqrt(np.mean(values.astype(np.float64) ** 2))) * 0.08)
    for start in starts:
        frame = values[start:start + width].astype(np.float64)
        result["time_seconds"].append(float((start + len(frame) / 2) / target_rate))
        frame -= np.mean(frame)
        voiced, frequency = False, None
        if len(frame) == width and np.sqrt(np.mean(frame ** 2)) >= gate:
            frame *= np.hanning(width)
            correlation = signal.correlate(frame, frame, mode="full", method="fft")[width - 1:]
            correlation /= max(float(correlation[0]), 1e-20)
            low, high = target_rate // 500, target_rate // 60
            candidates, _ = signal.find_peaks(correlation[low:high + 1])
            if len(candidates):
                lag = int(candidates[np.argmax(correlation[low:high + 1][candidates])]) + low
                if correlation[lag] >= 0.35:
                    left, center, right = correlation[lag - 1:lag + 2]
                    curvature = left - 2 * center + right
                    delta = 0.5 * (left - right) / curvature if abs(curvature) > 1e-12 else 0
                    frequency = float(target_rate / (lag + np.clip(delta, -0.5, 0.5)))
                    voiced = True
        result["f0_hz"].append(frequency)
        result["voiced"].append(voiced)
    return result


def analyze_audio(audio: Audio) -> dict:
    """Describe amplitude, approximate pitch and spectrum; no identity prediction."""
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise T2000Error("Audio diagnostics require finite, nonempty mono samples.")
    rate = audio.sample_rate
    if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)) or rate < 8000:
        raise T2000Error("Audio diagnostics require a valid sample rate of at least 8 kHz.")
    frequencies, power = signal.welch(samples.astype(np.float64), fs=rate, nperseg=min(2048, len(samples)))
    indices = np.unique(np.clip(np.searchsorted(
        frequencies, np.geomspace(max(20, frequencies[1]), rate / 2, 256)
    ), 0, len(frequencies) - 1)) if len(frequencies) > 1 else np.array([0])
    relative_db = 10 * np.log10(np.maximum(power[indices], 1e-20) / max(float(np.max(power)), 1e-20))
    return {
        "duration_seconds": len(samples) / rate, "sample_rate": int(rate),
        "rms": float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))),
        "peak": float(np.max(np.abs(samples))),
        "clipped_fraction": float(np.mean(np.abs(samples) >= 1)),
        "waveform": _waveform(samples, rate), "pitch": _pitch(samples, rate),
        "spectrum": {"method": "Mean power spectrum, relative dB; diagnostic only.",
                     "frequency_hz": frequencies[indices].astype(float).tolist(),
                     "relative_db": np.maximum(relative_db, -120).astype(float).tolist()},
        "preprocessing": {"version": PREPROCESSING_VERSION,
                          "channel_handling": "mean across source channels",
                          "sample_rate_handling": "native rate preserved; STT resamples separately",
                          "gain_handling": "original decoded amplitude preserved"},
    }

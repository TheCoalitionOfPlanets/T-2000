"""Bounded, deterministic delivery planning and cached reference diagnostics.

Planning preserves the submitted words and complete context when it fits. The
reference summary describes measured audio; it does not infer accent, emotion,
identity or perceptual naturalness, and does not modify the recording.
"""

from __future__ import annotations

from importlib import metadata
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter

import numpy as np

from .audio import analyze_audio, load_audio
from .errors import T2000Error
from .quality_models import QUALITY_MODELS, sha256, verified_model_path


PLANNER_VERSION = "context-delivery-plan-1"
PROFILE_VERSION = "reference-delivery-profile-1"
MAX_REQUEST_CHARACTERS = 10000
_F5_RATE, _F5_HOP = 24000, 256
_F5_REFERENCE_PADDING_SECONDS = 0.05
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "a.m.", "p.m.", "no.", "fig.", "inc.", "ltd.", "co.", "dept.",
    "approx.", "est.", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
    "sep.", "sept.", "oct.", "nov.", "dec.",
}
_CLOSING = '\"\u201d\u2019\u00bb\u203a\')]} '
_PAUSES = {"start": 0.0, "paragraph": 0.30, "sentence": 0.18, "clause": 0.08, "word": 0.08}
_BOUNDARY_PRIORITY = {"paragraph": 0, "sentence": 1, "clause": 2, "word": 3}
VAD_POLICY = {
    "sample_rate": 16000, "window_size": 512, "threshold": 0.5,
    "min_silence_duration": 0.15, "min_speech_duration": 0.1,
    "max_speech_duration": 30.0, "padding_seconds": 0.0, "num_threads": 2,
}


def _number(value, name, *, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise T2000Error(f"{name} must be finite and between {minimum:g} and {maximum:g}.")
    return float(value)


def _abbreviation(token):
    ending = token.rstrip(_CLOSING)
    return ending.lower() in _ABBREVIATIONS or re.fullmatch(r"(?:[A-Za-z]\.)+", ending) is not None


def _boundary(token, gap):
    if "\n" in gap or "\r" in gap:
        return "paragraph"
    ending = token.rstrip(_CLOSING)
    if ending.endswith(("!", "?")) or ending.endswith(".") and not _abbreviation(ending):
        return "sentence"
    return "clause" if ending.endswith((",", ";", ":", "—", "–")) else "word"


def plan_delivery(text, *, max_chars: int, reference_seconds: float | None = None,
                  reference_text: str = "", speed: float = 1.0,
                  max_context_seconds: float | None = None) -> dict:
    """Keep fitting context intact; split at meaningful boundaries only as needed.

    Offsets partition ``text.strip()`` exactly, including inter-segment whitespace.
    ``segment.text`` only strips its outer whitespace. When a context duration is
    supplied, fit uses the current F5 UTF-8 duration heuristic and 24 kHz/256-hop
    frame quantization. Original reference duration plus the worker's 50 ms
    conditioning tail is conservative after edge trimming. Pause values are
    metadata; this function inserts no audio or words.
    """
    started = perf_counter()
    if not isinstance(text, str) or not text.strip():
        raise T2000Error("Enter text to speak.")
    source = text.strip()
    if len(source) > MAX_REQUEST_CHARACTERS:
        raise T2000Error(f"Keep each request under {MAX_REQUEST_CHARACTERS} characters.")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1 <= max_chars <= 10000:
        raise T2000Error("The delivery segment limit must be an integer between 1 and 10000.")
    speed = _number(speed, "Speed", minimum=0.5, maximum=2.0)
    context = None
    reference_frames = reference_bytes = None
    if max_context_seconds is not None:
        context = _number(max_context_seconds, "Context duration", minimum=1, maximum=27)
        seconds = _number(reference_seconds, "Reference duration", minimum=1, maximum=12)
        if not isinstance(reference_text, str) or not reference_text.strip() or len(reference_text) > 2000:
            raise T2000Error("A matching reference transcript of at most 2000 characters is required.")
        prepared_text = reference_text.strip()
        prepared_text += " " if prepared_text.endswith((".", "!", "?")) else ". "
        reference_frames = int((seconds + _F5_REFERENCE_PADDING_SECONDS) * _F5_RATE) // _F5_HOP
        reference_bytes = len(prepared_text.encode("utf-8"))
        if reference_frames * _F5_HOP / _F5_RATE >= context:
            raise T2000Error("The reference leaves no output room in the model context.")
        max_chars = min(max_chars, 600)  # Current isolated F5 request protocol bound.

    def duration_bytes(text_bytes):
        if context is None:
            return None
        local_speed = speed * 0.3 if text_bytes < 10 else speed
        output_frames = int(reference_frames / reference_bytes * text_bytes / local_speed)
        return (reference_frames + output_frames) * _F5_HOP / _F5_RATE

    def duration(candidate):
        return duration_bytes(len(candidate.encode("utf-8")))

    def fits(candidate):
        estimate = duration(candidate)
        return len(candidate) <= max_chars and (estimate is None or (
            estimate <= context and estimate > reference_frames * _F5_HOP / _F5_RATE))

    constraints = {"max_chars": max_chars, "max_context_seconds": context,
                   "reference_seconds": reference_seconds if context is not None else None,
                   "reference_padding_reserve_seconds": _F5_REFERENCE_PADDING_SECONDS if context else 0.0,
                   "reference_frames": reference_frames, "reference_text_bytes": reference_bytes,
                   "speed": speed, "duration_method": "F5 UTF-8 reference-frame heuristic" if context else None}

    def segment(start, end, boundary, index, reason):
        utterance = source[start:end].strip()
        return {"text": utterance, "start": start, "end": end, "boundary": boundary,
                "pause_before_seconds": _PAUSES[boundary], "seed_offset": index,
                "reason": reason, "estimated_total_context_seconds": duration(utterance)}

    if fits(source):
        segments = [segment(0, len(source), "start", 0, "complete_context_fits")]
    else:
        words = list(re.finditer(r"\S+", source))
        if any(word.end() - word.start() > max_chars for word in words):
            raise T2000Error(f"A word exceeds the {max_chars}-character delivery limit. Shorten it or add spaces.")
        kinds = [_boundary(word.group(), source[word.end():words[index + 1].start()]
                           if index + 1 < len(words) else "") for index, word in enumerate(words)]
        abbreviations = [_abbreviation(word.group()) for word in words]
        byte_offsets = [0]
        for character in source:
            byte_offsets.append(byte_offsets[-1] + len(character.encode("utf-8")))
        # Backward reachability avoids an impossible short final batch: F5 slows
        # texts below ten UTF-8 bytes, so context fit is not strictly monotonic.
        choices = {}
        reachable = {len(words)}
        for first in range(len(words) - 1, -1, -1):
            candidates = []
            start = words[first].start()
            for last in range(first, len(words)):
                if words[last].end() - start > max_chars:
                    break
                if last + 1 not in reachable:
                    continue
                estimate = duration_bytes(byte_offsets[words[last].end()] - byte_offsets[start])
                if estimate is None or (estimate <= context
                                        and estimate > reference_frames * _F5_HOP / _F5_RATE):
                    abbreviation_penalty = int(abbreviations[last] and last + 1 < len(words))
                    priority = -1 if last + 1 == len(words) else _BOUNDARY_PRIORITY[kinds[last]]
                    candidates.append(((abbreviation_penalty, priority, -last), last))
            if candidates:
                choices[first] = min(candidates)[1]
                reachable.add(first)
        if 0 not in reachable:
            raise T2000Error("The text cannot fit this reference's model context without splitting a word. "
                             "Use shorter text, a matching reference transcript, or faster delivery.")
        segments, first, preceding = [], 0, "start"
        while first < len(words):
            cut = choices[first]
            end = words[cut + 1].start() if cut + 1 < len(words) else len(source)
            segments.append(segment(words[first].start(), end, preceding, len(segments),
                                    "context_limit" if context else "character_limit"))
            preceding, first = kinds[cut], cut + 1
    return {"version": PLANNER_VERSION, "source_text": source, "segments": segments,
            "constraints": constraints, "planning_seconds": perf_counter() - started,
            "preserves_source_text": True, "pause_policy": "metadata only; no samples inserted",
            "style_decision": "Preserve reference conditioning and sentence context; no emotion inference."}


def _speech_intervals(audio, vad_path):
    """Unpadded CPU VAD intervals; pause analysis must not remove internal gaps."""
    import sherpa_onnx
    import soxr

    policy = VAD_POLICY
    rate = policy["sample_rate"]
    detection = (audio.samples if audio.sample_rate == rate else
                 soxr.resample(audio.samples, audio.sample_rate, rate, quality="HQ"))
    detection = np.ascontiguousarray(detection, dtype=np.float32)
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_path)
    for name in ("threshold", "min_silence_duration", "min_speech_duration", "max_speech_duration", "window_size"):
        setattr(config.silero_vad, name, policy[name])
    config.sample_rate, config.num_threads, config.provider = rate, policy["num_threads"], "cpu"
    detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=35)
    intervals = []

    def drain():
        while not detector.empty():
            part = detector.front
            start = max(0.0, float(part.start / rate))
            end = min(audio.duration_seconds, float((part.start + len(part.samples)) / rate))
            if end > start:
                intervals.append((start, end))
            detector.pop()

    window = policy["window_size"]
    for offset in range(0, detection.size, window):
        block = detection[offset:offset + window]
        detector.accept_waveform(np.pad(block, (0, window - len(block))) if len(block) < window else block)
        drain()
    detector.flush()
    drain()
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _quantiles(values):
    return {name: float(np.percentile(values, percentile)) if values else None
            for name, percentile in (("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90))}


def reference_delivery_profile(reference_path, transcript, model_dir, cache_dir):
    """Cache measured reference delivery, without model loading or hidden downloads.

    Complete input is limited to 1–30 seconds. A missing or invalid local VAD
    raises an actionable error; callers may explicitly use a simpler fallback.
    No detected speech produces unknown rates, rather than invented delivery.
    """
    started = perf_counter()
    if not isinstance(transcript, str) or len(transcript) > 10000:
        raise T2000Error("Use a reference transcript of at most 10000 characters.")
    reference_path, cache_dir = Path(reference_path), Path(cache_dir)
    try:
        if reference_path.stat().st_size > 256 * 1024 * 1024:
            raise T2000Error("The reference file is too large for bounded delivery analysis.")
    except OSError as exc:
        raise T2000Error(f"Could not read the delivery reference: {exc}") from exc
    # Validate bounds on every read, including cache hits.
    import soundfile as sf

    try:
        info = sf.info(reference_path)
        if not 1 <= info.duration <= 30 or not 8000 <= info.samplerate <= 192000 or not 1 <= info.channels <= 8:
            raise T2000Error("Delivery analysis requires a complete 1–30 second reference WAV.")
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, T2000Error):
            raise
        raise T2000Error(f"Could not read the delivery reference: {exc}") from exc
    vad_path = verified_model_path(model_dir, "silero_vad.onnx")
    versions = {}
    for package in ("sherpa-onnx", "numpy", "scipy", "soxr"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    identity = {"version": PROFILE_VERSION, "audio_sha256": sha256(reference_path),
                "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                "vad_sha256": QUALITY_MODELS["silero_vad.onnx"]["sha256"],
                "vad_policy": VAD_POLICY, "runtime_versions": versions}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_path = cache_dir / f"delivery-{key}.json"
    if cache_path.is_file() and cache_path.stat().st_size <= 1024 * 1024:
        try:
            saved = json.loads(cache_path.read_text())
            if (saved.get("cache_key") == key and saved.get("identity") == identity
                    and isinstance(saved.get("pace"), dict) and isinstance(saved.get("pitch"), dict)
                    and isinstance(saved.get("pauses"), dict)):
                json.dumps(saved, allow_nan=False)
                return dict(saved, cache_hit=True, profile_seconds=perf_counter() - started)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    audio = load_audio(reference_path)
    analysis = analyze_audio(audio)
    intervals = _speech_intervals(audio, vad_path)
    duration = audio.duration_seconds
    # Validate detector boundaries before deriving rates or indexing samples.
    if any(not (0 <= start < end <= duration) for start, end in intervals):
        raise T2000Error("The voice activity detector returned invalid speech boundaries.")
    speech_seconds = sum(end - start for start, end in intervals)
    pauses = [second[0] - first[1] for first, second in zip(intervals, intervals[1:])]
    if any(value < 0 for value in pauses):
        raise T2000Error("The voice activity detector returned overlapping speech boundaries.")
    word_count = len(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", transcript, re.UNICODE))
    has_speech = speech_seconds > 0
    overall_rate = word_count / duration if word_count and has_speech else None
    active_rate = word_count / speech_seconds if word_count and has_speech else None
    usable_pace = bool(word_count >= 5 and duration >= 2 and speech_seconds >= 1
                       and active_rate is not None and 0.5 <= active_rate <= 8
                       and speech_seconds / duration >= 0.2)
    pitches = [value for value in analysis["pitch"]["f0_hz"]
               if value is not None and math.isfinite(value) and value > 0]
    active_samples = [audio.samples[round(start * audio.sample_rate):round(end * audio.sample_rate)]
                      for start, end in intervals]
    active_energy = sum(float(np.sum(part.astype(np.float64) ** 2)) for part in active_samples)
    active_frames = sum(part.size for part in active_samples)
    result = {
        "version": PROFILE_VERSION, "cache_key": key, "identity": identity,
        "duration_seconds": duration, "sample_rate": audio.sample_rate,
        "speech_detected": has_speech, "speech_seconds": speech_seconds,
        "speech_fraction": speech_seconds / duration,
        "speech_intervals_seconds": [[start, end] for start, end in intervals],
        "pace": {"transcript_words": word_count, "words_per_second_including_pauses": overall_rate,
                 "words_per_second_during_detected_speech": active_rate,
                 "words_per_minute_including_pauses": overall_rate * 60 if overall_rate is not None else None,
                 "usable_for_control": usable_pace, "requires_transcript_review": True,
                 "method": "Supplied transcript word count divided by clip duration and unpadded VAD speech duration",
                 "guard": "At least five words, two clip seconds, one speech second; plausible articulation rate."},
        "pauses": {"internal_count": len(pauses), "internal_seconds": pauses,
                   "quantiles_seconds": _quantiles(pauses),
                   "leading_seconds": intervals[0][0] if has_speech else None,
                   "trailing_seconds": duration - intervals[-1][1] if has_speech else None,
                   "internal_fraction": sum(pauses) / duration if has_speech else None,
                   "method": "Unpadded VAD gaps of roughly 150 ms or longer; no word alignment or punctuation inference."},
        "pitch": {"median_hz": float(np.median(pitches)) if pitches and has_speech else None,
                  "iqr_hz": float(np.percentile(pitches, 75) - np.percentile(pitches, 25))
                  if pitches and has_speech else None,
                  "voiced_diagnostic_frames": len(pitches),
                  "diagnostic_frames": len(analysis["pitch"]["f0_hz"]),
                  "method": analysis["pitch"]["method"]},
        "level": {"rms": analysis["rms"], "peak": analysis["peak"],
                  "clipped_fraction": analysis["clipped_fraction"],
                  "speech_rms": math.sqrt(active_energy / active_frames) if active_frames else None,
                  "method": "Decoded recording level, not perceived loudness or an emotion estimate."},
        "accent": {"status": "unknown", "reason": "Not inferred from pitch, spectrum or a short clip."},
        "emotion": {"status": "unknown", "reason": "Not inferred from pitch, volume or punctuation."},
        "limitations": ["Transcript rates require correct supplied words; no automatic review is implied.",
                        "F0 and pause estimates can be wrong; they do not measure identity or naturalness.",
                        "No accent, emotion, exact inflection or habitual delivery transfer is established."],
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".delivery-", suffix=".json", dir=cache_dir)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(json.dumps(result, allow_nan=False, indent=2) + "\n")
        os.replace(temporary, cache_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dict(result, cache_hit=False, profile_seconds=perf_counter() - started)

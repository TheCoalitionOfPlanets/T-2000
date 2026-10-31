"""Offline objective proxies for matched voice experiments, never a quality verdict."""

import math
from pathlib import Path
import statistics
import unicodedata

import numpy as np
import soundfile as sf
import soxr

from .audio import Audio, analyze_audio
from .errors import T2000Error
from .quality_models import sha256, verified_model_path


SPEAKER_MODEL_NAME = "wespeaker_en_voxceleb_resnet34.onnx"


def normalized_text(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(char for char in text
                            if not unicodedata.category(char).startswith(("P", "S"))).split())


def edit_counts(reference, hypothesis):
    """Deterministic Levenshtein ties prefer substitution, deletion, insertion."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                current.append(previous[j - 1])
                continue
            sub, delete, insert = previous[j - 1], previous[j], current[j - 1]
            current.append(min((
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ), key=lambda value: value[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {"errors": errors, "substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "reference_units": len(reference),
            "rate": errors / len(reference) if reference else None}


def error_metrics(expected, recognized):
    reference, hypothesis = normalized_text(expected), normalized_text(recognized)
    if not reference:
        raise ValueError("Expected speech text must contain a scored word.")
    return {"expected_normalized": reference, "recognized_normalized": hypothesis,
            "word": edit_counts(reference.split(), hypothesis.split()),
            "character": edit_counts(reference.replace(" ", ""), hypothesis.replace(" ", ""))}


def audio_metrics(path):
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    finite = bool(np.isfinite(samples).all())
    frames, channels = samples.shape
    peak = float(np.max(np.abs(samples))) if samples.size and finite else None
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size and finite else None
    # PCM16 WAV export saturates positive values at 32767/32768.
    clipped = int(np.count_nonzero(np.abs(samples) >= 32767 / 32768)) if finite else None
    return {"sample_rate": int(rate), "frames": int(frames), "channels": int(channels),
            "duration_seconds": frames / rate, "finite": finite, "peak": peak, "rms": rms,
            "clipped_samples": clipped,
            "clipped_fraction": clipped / samples.size if samples.size and finite else None,
            "valid": bool(frames > 0 and channels == 1 and finite and rms > 0),
            "wav_sha256": sha256(Path(path))}


def descriptive_features(path, recognized_text):
    """Approximate pitch/spectrum and transcript pace, without a perceptual quality score."""
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    analysis = analyze_audio(Audio(samples.mean(axis=1), rate))
    pitch = analysis["pitch"]
    values = [value for value in pitch["f0_hz"] if value is not None]
    steps = [abs(12 * math.log2(second / first)) for first, second
             in zip(pitch["f0_hz"], pitch["f0_hz"][1:]) if first and second]
    duration = analysis["duration_seconds"]
    words = len(normalized_text(recognized_text).split())
    return {
        "pitch": {"method": pitch["method"], "median_hz": float(np.median(values)) if values else None,
                  "iqr_hz": float(np.percentile(values, 75) - np.percentile(values, 25)) if values else None,
                  "voiced_diagnostic_frames": len(values), "diagnostic_frames": len(pitch["f0_hz"]),
                  "median_absolute_frame_step_semitones": statistics.median(steps) if steps else None},
        "spectrum": analysis["spectrum"],
        "pace": {"recognized_words": words, "full_duration_seconds": duration,
                 "words_per_second_including_pauses": words / duration if duration else None,
                 "speech_only_duration_seconds": None,
                 "method": "ASR word count / full clip duration including pauses; no speech-only rate inferred"},
        "limitation": "Pitch, spectral shape and pitch-step diagnostics describe audio; "
                      "they are not speaker identity, naturalness or smoothness quality scores.",
    }


class SpeakerScorer:
    """Independent WeSpeaker ONNX embeddings on CPU, with no implicit downloads."""

    def __init__(self, model_dir, num_threads=2):
        import sherpa_onnx

        self.path = verified_model_path(model_dir, SPEAKER_MODEL_NAME)
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(self.path), num_threads=num_threads, debug=False, provider="cpu",
        )
        if not config.validate():
            raise T2000Error("Invalid local speaker-verification model configuration.")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.cache = {}

    def embedding(self, path):
        key = sha256(Path(path))
        if key in self.cache:
            return self.cache[key]
        samples, rate = sf.read(path, dtype="float64", always_2d=True)
        if not samples.size or not np.isfinite(samples).all():
            raise ValueError(f"Empty or non-finite speaker audio: {path}")
        mono = samples.mean(axis=1).astype(np.float32)
        if not np.any(mono):
            raise ValueError(f"Silent speaker audio: {path}")
        if rate != 16000:
            mono = soxr.resample(mono, rate, 16000, quality="HQ")
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=np.ascontiguousarray(mono, dtype=np.float32))
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            raise ValueError(f"Speaker audio is too short for the independent encoder: {path}")
        vector = np.asarray(self.extractor.compute(stream), dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not vector.size or not np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid independent speaker embedding")
        vector /= norm
        self.cache[key] = vector
        return vector

    def cosine(self, reference, generated):
        return float(np.clip(np.dot(self.embedding(reference), self.embedding(generated)), -1, 1))


def eligible(row):
    audio = row.get("audio", {})
    metrics = row.get("metrics", {})
    rate = audio.get("sample_rate", 0)
    expected_rate = row.get("backend_sample_rate")
    return bool(row.get("complete") is True and row.get("score_status") == "ok"
                and audio.get("valid") is True and 8000 <= rate <= 96000
                and (expected_rate is None or rate == expected_rate)
                and audio.get("clipped_samples") == 0 and metrics.get("clipped_fraction", 0) == 0)


def summarize(rows):
    scored = [row for row in rows if row.get("score_status") == "ok"]
    asr_scored = [row for row in rows if row.get("asr") is not None]
    valid = [row for row in rows if eligible(row)]
    result = {"requests": len(rows), "generated": sum(row.get("complete") is True for row in rows),
              "generation_failures": sum(row.get("complete") is not True for row in rows),
              "scored": len(scored), "score_failures": len(rows) - len(scored), "eligible": len(valid),
              "asr_scored": len(asr_scored), "asr_failures": len(rows) - len(asr_scored)}
    for unit in ("word", "character"):
        totals = {key: sum(row["asr"][unit][key] for row in asr_scored)
                  for key in ("errors", "substitutions", "deletions", "insertions", "reference_units")}
        totals["rate"] = totals["errors"] / totals["reference_units"] if totals["reference_units"] else None
        result[unit] = totals
    for key in ("speaker_cosine", "generation_seconds", "first_pcm_seconds", "rtf", "audio_seconds"):
        values = [row.get(key, row.get("metrics", {}).get(key)) for row in valid]
        values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        result[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)} \
            if values else None
    return result


def paired_comparisons(rows, baseline):
    """Match exact reference hash/case/seed; retain incomplete pairs and duration changes."""
    grouped = {}
    variants = sorted({row["variant"] for row in rows} - {baseline})
    for row in rows:
        key = (row["reference_sha256"], row["case"], row["seed"])
        group = grouped.setdefault(key, {})
        if row["variant"] in group:
            raise ValueError(f"Duplicate matched request: {key}, {row['variant']}")
        group[row["variant"]] = row
    pairs = []
    for (reference, case, seed), group in sorted(grouped.items()):
        control = group.get(baseline)
        for variant in variants:
            candidate = group.get(variant)
            pair = {"reference_sha256": reference, "case": case, "seed": seed,
                    "baseline": baseline, "variant": variant, "eligible": False}
            if not control or not candidate:
                pair["exclusion_reason"] = "Missing baseline or candidate request."
            elif control["text"] != candidate["text"] or control["expected_text"] != candidate["expected_text"]:
                pair["exclusion_reason"] = "Input text or expected transcript differs."
            elif not (eligible(control) and eligible(candidate)):
                pair["exclusion_reason"] = "Failed, invalid, clipped, incomplete, or unscored request."
            else:
                ratio = candidate["audio"]["duration_seconds"] / control["audio"]["duration_seconds"]
                pair.update(
                    eligible=True,
                    speaker_cosine_delta=candidate["speaker_cosine"] - control["speaker_cosine"],
                    word_error_delta=candidate["asr"]["word"]["errors"] - control["asr"]["word"]["errors"],
                    candidate_over_baseline_duration=ratio,
                    baseline_frames=control["audio"]["frames"], candidate_frames=candidate["audio"]["frames"],
                    duration_change_over_ten_percent=abs(ratio - 1) > 0.1,
                    assessment="Objective proxies only; review duration, transcript and paired listening.",
                )
            pairs.append(pair)
    return pairs

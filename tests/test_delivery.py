"""Context preservation and measured reference delivery without model downloads."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import soundfile as sf

from t2000 import delivery
from t2000.delivery import plan_delivery, reference_delivery_profile
from t2000.errors import T2000Error


def test_fitting_passage_preserves_paragraphs_sentence_context_and_seed():
    text = '  Dr. Smith said, "Hello there!"  Please wait.\n\nA second paragraph.  '
    result = plan_delivery(text, max_chars=600)
    assert len(result["segments"]) == 1
    part = result["segments"][0]
    assert part["text"] == text.strip()
    assert part["start"] == 0 and part["end"] == len(text.strip())
    assert part["boundary"] == "start" and part["seed_offset"] == 0
    assert part["pause_before_seconds"] == 0
    assert result["preserves_source_text"] and result["planning_seconds"] >= 0
    assert "no emotion inference" in result["style_decision"]


def test_needed_splits_preserve_exact_source_offsets_words_and_natural_boundaries():
    text = 'Dr. Smith paid 12.50 dollars.\n\nHe saw J. Brown;  they waved. "Good morning!" They left.'
    result = plan_delivery(text, max_chars=40)
    parts = result["segments"]
    assert len(parts) > 1
    assert "".join(text[part["start"]:part["end"]] for part in parts) == text
    assert " ".join(part["text"] for part in parts).split() == text.split()
    assert all(len(part["text"]) <= 40 for part in parts)
    assert "Dr. Smith" in parts[0]["text"] and "12.50" in parts[0]["text"]
    assert parts[1]["boundary"] == "paragraph" and parts[1]["pause_before_seconds"] == 0.30
    assert [part["seed_offset"] for part in parts] == list(range(len(parts)))
    assert all(not part["text"].endswith(("Dr.", "J.")) for part in parts)


@pytest.mark.parametrize("text", ["Hello! Next sentence? Last one.", "Mr. Brown met Dr. Green. Then left.",
                                  "One paragraph.\n\nAnother paragraph with words.",
                                  "The price is 1.25 and the time is 9.30 a.m. today.",
                                  "One  two\tthree\n\nfour five six seven eight nine ten eleven twelve."])
def test_partition_preserves_text_at_several_limits(text):
    for maximum in (18, 25, 40):
        plan = plan_delivery(text, max_chars=maximum)
        source = text.strip()
        assert "".join(source[p["start"]:p["end"]] for p in plan["segments"]) == source
        assert all(p["text"] == source[p["start"]:p["end"]].strip() for p in plan["segments"])
        assert all(len(p["text"]) <= maximum for p in plan["segments"])
        assert " ".join(p["text"] for p in plan["segments"]).split() == source.split()


def worker_module():
    path = Path(__file__).resolve().parents[1] / "scripts/f5_infer_worker.py"
    spec = importlib.util.spec_from_file_location("delivery_f5_context", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f5_planning_matches_actual_worker_frame_budget_and_keeps_adjacent_sentences():
    worker = worker_module()
    reference_seconds = 4.35
    transcript = "These are the exact words from the reference recording."
    text = "The first sentence has useful context. The next sentence belongs with it."
    plan = plan_delivery(text, max_chars=600, reference_seconds=reference_seconds,
                         reference_text=transcript, speed=1.0, max_context_seconds=27)
    assert len(plan["segments"]) == 1
    frames = int((reference_seconds + 0.05) * worker.SAMPLE_RATE) // worker.HOP_LENGTH
    prepared = transcript + " "
    actual = worker.estimated_duration_frames(frames, prepared, text, 1.0) * worker.HOP_LENGTH / worker.SAMPLE_RATE
    assert plan["segments"][0]["estimated_total_context_seconds"] == actual
    assert plan["constraints"]["reference_padding_reserve_seconds"] == 0.05
    long_text = (text + " ") * 10
    plan = plan_delivery(long_text, max_chars=600, reference_seconds=reference_seconds,
                         reference_text=transcript, speed=0.7, max_context_seconds=27)
    assert len(plan["segments"]) > 1
    assert " ".join(part["text"] for part in plan["segments"]).split() == long_text.split()
    assert all(worker.estimated_duration_frames(frames, prepared, part["text"], 0.7)
               * worker.HOP_LENGTH / worker.SAMPLE_RATE <= 27 for part in plan["segments"])
    assert all(len(part["text"]) <= 600 for part in plan["segments"])


def test_short_text_slowdown_does_not_reject_a_longer_fitting_sentence():
    # A five-byte word alone needs more context than that word plus another:
    # the worker's sub-ten-byte slowdown makes naive monotonic fit checks wrong.
    plan = plan_delivery("Hello world. Hello world.", max_chars=12, reference_seconds=12,
                         reference_text="Reference.", speed=1, max_context_seconds=27)
    assert [part["text"] for part in plan["segments"]] == ["Hello world.", "Hello world."]
    with pytest.raises(T2000Error, match="cannot fit"):
        plan_delivery("Hello", max_chars=600, reference_seconds=12,
                      reference_text="Reference.", speed=1, max_context_seconds=27)


@pytest.mark.parametrize("kwargs", [
    {"text": "", "max_chars": 100}, {"text": None, "max_chars": 100},
    {"text": "hello", "max_chars": True}, {"text": "hello", "max_chars": 0},
    {"text": "hello", "max_chars": 10, "speed": float("nan")},
    {"text": "hello", "max_chars": 10, "speed": True},
    {"text": "hello", "max_chars": 10, "max_context_seconds": 27},
    {"text": "hello", "max_chars": 10, "max_context_seconds": 27, "reference_seconds": 13,
     "reference_text": "hello reference"},
    {"text": "hello", "max_chars": 10, "max_context_seconds": 27, "reference_seconds": 4,
     "reference_text": ""},
])
def test_invalid_planning_inputs_are_actionable(kwargs):
    with pytest.raises(T2000Error):
        plan_delivery(**kwargs)


def test_unbroken_token_is_rejected_instead_of_corrupted():
    with pytest.raises(T2000Error, match="word exceeds"):
        plan_delivery("a" * 50, max_chars=20)


@pytest.fixture
def reference(tmp_path, monkeypatch):
    path = tmp_path / "reference.wav"
    rate = 16000
    samples = np.sin(2 * np.pi * 180 * np.arange(rate * 4) / rate).astype(np.float32) * 0.1
    sf.write(path, samples, rate, subtype="FLOAT")
    original = path.read_bytes()
    calls = []
    monkeypatch.setattr(delivery, "verified_model_path", lambda root, name: tmp_path / name)

    def detect(audio, model):
        calls.append((audio.duration_seconds, model))
        return [(0.25, 1.25), (1.75, 3.5)]

    monkeypatch.setattr(delivery, "_speech_intervals", detect)
    return path, original, calls, tmp_path / "models", tmp_path / "cache"


def test_cached_reference_profile_has_measured_rates_pauses_pitch_and_unknown_style(reference):
    path, original, calls, models, cache = reference
    transcript = "These are the exact eight words spoken here"
    result = reference_delivery_profile(path, transcript, models, cache)
    assert not result["cache_hit"] and len(calls) == 1
    assert result["speech_seconds"] == pytest.approx(2.75)
    assert result["pace"]["transcript_words"] == 8
    assert result["pace"]["words_per_second_including_pauses"] == pytest.approx(8 / 4)
    assert result["pace"]["words_per_second_during_detected_speech"] == pytest.approx(8 / 2.75)
    assert result["pace"]["usable_for_control"] and result["pace"]["requires_transcript_review"]
    assert result["pauses"]["internal_seconds"] == [0.5]
    assert result["pauses"]["leading_seconds"] == 0.25
    assert result["pauses"]["trailing_seconds"] == 0.5
    assert result["pauses"]["quantiles_seconds"]["p50"] == 0.5
    assert result["pitch"]["median_hz"] == pytest.approx(180, rel=0.05)
    assert result["level"]["rms"] > 0 and result["level"]["speech_rms"] > 0
    assert result["accent"]["status"] == result["emotion"]["status"] == "unknown"
    hit = reference_delivery_profile(path, transcript, models, cache)
    assert hit["cache_hit"] and len(calls) == 1
    assert hit["cache_key"] == result["cache_key"]
    changed = reference_delivery_profile(path, transcript + " changed", models, cache)
    assert not changed["cache_hit"] and changed["cache_key"] != result["cache_key"] and len(calls) == 2
    assert path.read_bytes() == original
    assert not list(cache.glob(".delivery-*"))


def test_empty_detection_never_invents_pace_pause_or_pitch(reference, monkeypatch):
    path, _, _, models, cache = reference
    monkeypatch.setattr(delivery, "_speech_intervals", lambda audio, model: [])
    result = reference_delivery_profile(path, "These are several supplied words", models, cache)
    assert not result["speech_detected"] and result["speech_seconds"] == 0
    assert result["pace"]["words_per_second_including_pauses"] is None
    assert result["pace"]["words_per_second_during_detected_speech"] is None
    assert not result["pace"]["usable_for_control"]
    assert result["pauses"]["leading_seconds"] is None
    assert result["pauses"]["quantiles_seconds"]["p50"] is None
    assert result["pitch"]["median_hz"] is None


def test_cache_audio_change_corruption_and_invalid_detector_are_not_trusted(reference, monkeypatch):
    path, _, calls, models, cache = reference
    first = reference_delivery_profile(path, "One two three four five six", models, cache)
    cached = next(cache.glob("delivery-*.json"))
    cached.write_text("{broken")
    assert not reference_delivery_profile(path, "One two three four five six", models, cache)["cache_hit"]
    assert len(calls) == 2
    samples, rate = sf.read(path)
    sf.write(path, samples * 0.5, rate, subtype="FLOAT")
    second = reference_delivery_profile(path, "One two three four five six", models, cache)
    assert first["cache_key"] != second["cache_key"]
    monkeypatch.setattr(delivery, "_speech_intervals", lambda audio, model: [(0, 5)])
    with pytest.raises(T2000Error, match="invalid speech boundaries"):
        reference_delivery_profile(path, "Changed transcript to miss cache", models, cache)


def test_reference_limits_missing_vad_and_no_words_are_explicit(reference, monkeypatch):
    path, _, _, models, cache = reference
    with pytest.raises(T2000Error, match="transcript"):
        reference_delivery_profile(path, None, models, cache)
    with pytest.raises(T2000Error, match="Could not read"):
        reference_delivery_profile(path.parent / "missing.wav", "words", models, cache)
    result = reference_delivery_profile(path, "", models, cache)
    assert result["pace"]["transcript_words"] == 0
    assert result["pace"]["words_per_second_including_pauses"] is None
    sf.write(path, np.zeros(16000 // 2), 16000)
    with pytest.raises(T2000Error, match="1–30"):
        reference_delivery_profile(path, "words", models, cache)
    sf.write(path, np.full(16000 * 31, 0.1), 16000)
    with pytest.raises(T2000Error, match="1–30"):
        reference_delivery_profile(path, "words", models, cache)
    sf.write(path, np.full(16000, 0.1), 16000)
    monkeypatch.setattr(delivery, "verified_model_path", lambda *args: (_ for _ in ()).throw(T2000Error("missing VAD")))
    with pytest.raises(T2000Error, match="missing VAD"):
        reference_delivery_profile(path, "words", models, cache)


def test_vad_uses_cpu_unpadded_intervals_and_clamps_final_padded_window(monkeypatch):
    from t2000.audio import Audio

    calls = []
    queue = []

    class Detector:
        def __init__(self, config, buffer_size_in_seconds):
            calls.append(config)
            assert buffer_size_in_seconds == 35

        def accept_waveform(self, block):
            assert block.size == 512

        def empty(self):
            return not queue

        @property
        def front(self):
            return queue[0]

        def pop(self):
            queue.pop(0)

        def flush(self):
            queue.extend([SimpleNamespace(start=1600, samples=np.zeros(3200)),
                          SimpleNamespace(start=16000 - 200, samples=np.zeros(512))])

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        VadModelConfig=lambda: SimpleNamespace(silero_vad=SimpleNamespace()), VoiceActivityDetector=Detector))
    result = delivery._speech_intervals(Audio(np.ones(16000, dtype=np.float32), 16000), Path("pinned-vad"))
    assert result == [(0.1, 0.3), ((16000 - 200) / 16000, 1.0)]
    assert calls[0].provider == "cpu" and calls[0].num_threads == 2
    assert calls[0].silero_vad.model == "pinned-vad"
    assert calls[0].silero_vad.min_silence_duration == 0.15

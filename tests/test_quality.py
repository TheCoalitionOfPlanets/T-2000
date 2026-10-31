"""Meaningful offline checks of scoring, exact pairing and pinned model publication."""

import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import soundfile as sf

from t2000.errors import T2000Error
from t2000.quality import (SpeakerScorer, audio_metrics, descriptive_features, eligible, error_metrics,
                          paired_comparisons, summarize)
import t2000.quality_models as models


def test_transcript_counts_preserve_numbers_and_count_insertions_deletions_and_empty_output():
    scores = error_metrics("Hello, ＷＯＲＬＤ!", "hello world")
    assert scores["word"]["errors"] == 0
    assert error_metrics("one two three", "one four")["word"] == {
        "errors": 2, "substitutions": 1, "deletions": 1, "insertions": 0, "reference_units": 3, "rate": 2 / 3,
    }
    assert error_metrics("one", "one two three")["word"]["rate"] == 2  # WER may exceed one.
    assert error_metrics("one two", "")["word"]["deletions"] == 2
    assert error_metrics("42", "forty two")["word"]["errors"] > 0
    with pytest.raises(ValueError, match="scored word"):
        error_metrics("...", "hello")


def test_audio_checks_reject_zero_nonfinite_and_pcm16_saturation(tmp_path):
    path = tmp_path / "test.wav"
    sf.write(path, np.zeros(16000), 16000)
    assert not audio_metrics(path)["valid"]
    sf.write(path, np.full(16000, 1.0), 16000, subtype="PCM_16")
    clipped = audio_metrics(path)
    assert clipped["clipped_samples"] == 16000
    sf.write(path, np.array([0.1, float("nan")]), 16000, subtype="FLOAT")
    assert not audio_metrics(path)["finite"]
    with pytest.raises((OSError, RuntimeError)):
        audio_metrics(tmp_path / "missing.wav")


def row(variant="baseline", *, reference="hash1", case="case1", seed=42, duration=2, text="hello world"):
    return {"variant": variant, "reference_sha256": reference, "case": case, "seed": seed,
            "text": text, "expected_text": text, "complete": True, "score_status": "ok",
            "speaker_cosine": 0.5, "metrics": {"clipped_fraction": 0, "audio_seconds": duration,
                                               "generation_seconds": 0.2, "rtf": 0.1},
            "audio": {"valid": True, "sample_rate": 22050, "clipped_samples": 0,
                      "duration_seconds": duration, "frames": int(duration * 22050)},
            "asr": error_metrics(text, text)}


def test_pairs_require_exact_reference_case_seed_and_count_missing_baseline():
    rows = [row(), row("candidate"), row("candidate", reference="hash2"), row("candidate", seed=43)]
    pairs = paired_comparisons(rows, "baseline")
    assert len(pairs) == 3
    assert sum(pair["eligible"] for pair in pairs) == 1
    assert sum("Missing" in pair.get("exclusion_reason", "") for pair in pairs) == 2
    with pytest.raises(ValueError, match="Duplicate"):
        paired_comparisons(rows + [row()], "baseline")


def test_pairs_exclude_wrong_text_and_failures_without_hiding_duration_changes():
    control, candidate = row(), row("candidate", duration=1)
    candidate["speaker_cosine"] = 0.9
    pair = paired_comparisons([control, candidate], "baseline")[0]
    assert pair["eligible"] and pair["duration_change_over_ten_percent"]
    assert pair["candidate_over_baseline_duration"] == 0.5
    assert "winner" not in pair
    candidate["text"] = "different"
    assert not paired_comparisons([control, candidate], "baseline")[0]["eligible"]
    candidate["text"] = control["text"]
    candidate["complete"] = False
    assert not paired_comparisons([control, candidate], "baseline")[0]["eligible"]


def test_scored_clipped_outputs_keep_word_errors_but_do_not_enter_eligible_medians():
    good, clipped, failed = row(), row("candidate"), row("failed")
    clipped["audio"]["clipped_samples"] = 1
    clipped["asr"] = error_metrics("hello world", "hello")
    failed.update(complete=False, score_status="error")
    failed.pop("asr")  # A generation failure has no output to transcribe.
    summary = summarize([good, clipped, failed])
    assert summary["requests"] == 3 and summary["generated"] == 2 and summary["eligible"] == 1
    assert summary["word"]["errors"] == 1 and summary["word"]["reference_units"] == 4
    assert not eligible(clipped)
    good["audio"]["sample_rate"] = 24000  # Future direct backends need no sample-rate-specific score gate.
    good["backend_sample_rate"] = 24000
    assert eligible(good)
    good["backend_sample_rate"] = 22050
    assert not eligible(good)
    good.pop("backend_sample_rate")
    for invalid_rate in (7999, 96001):
        good["audio"]["sample_rate"] = invalid_rate
        assert not eligible(good)


def test_failed_speaker_verification_does_not_hide_completed_asr_errors():
    failed = row()
    failed.update(score_status="error", score_error="Speaker embedding too short")
    failed["asr"] = error_metrics("hello world", "")
    summary = summarize([failed])
    assert summary["scored"] == 0 and summary["asr_scored"] == 1 and summary["eligible"] == 0
    assert summary["word"]["errors"] == 2 and summary["word"]["rate"] == 1
    assert summary["speaker_cosine"] is None


def test_speaker_extraction_is_full_mono_16k_normalized_cached_and_rejects_silence(tmp_path, monkeypatch):
    import t2000.quality as quality

    calls = []

    class Stream:
        def accept_waveform(self, *, sample_rate, waveform):
            calls.append((sample_rate, waveform.copy()))

        def input_finished(self):
            pass

    class Extractor:
        def __init__(self, config):
            assert config.provider == "cpu"

        def create_stream(self):
            return Stream()

        def is_ready(self, stream):
            return True

        def compute(self, stream):
            return [3, 4]

    module = SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=lambda **kwargs: SimpleNamespace(**kwargs, validate=lambda: True),
        SpeakerEmbeddingExtractor=Extractor,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    monkeypatch.setattr(quality, "verified_model_path", lambda root, name: tmp_path / name)
    path = tmp_path / "voice.wav"
    sf.write(path, np.full((24000, 2), 0.1), 24000, subtype="FLOAT")
    scorer = SpeakerScorer(tmp_path)
    np.testing.assert_allclose(scorer.embedding(path), [0.6, 0.8])
    assert scorer.cosine(path, path) == pytest.approx(1)
    assert len(calls) == 1 and calls[0][0] == 16000 and calls[0][1].shape == (16000,)
    sf.write(path, np.zeros(24000), 24000)
    with pytest.raises(ValueError, match="Silent"):
        scorer.embedding(path)


def test_pitch_and_reference_pace_are_descriptive_and_include_pauses(tmp_path):
    path = tmp_path / "voice.wav"
    samples = np.sin(2 * np.pi * 180 * np.arange(32000) / 16000) * 0.1
    sf.write(path, samples, 16000, subtype="FLOAT")
    features = descriptive_features(path, "four words spoken here")
    assert features["pace"]["words_per_second_including_pauses"] == 2
    assert features["pitch"]["median_hz"] == pytest.approx(180, rel=0.05)
    assert "not speaker identity" in features["limitation"]


@pytest.fixture
def pinned_model(tmp_path, monkeypatch):
    payload = b"known model bytes"
    spec = {"url": "https://example.invalid/model", "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(models, "QUALITY_MODELS", {"model.onnx": spec})
    return payload, tmp_path / "quality/model.onnx"


def test_model_integrity_missing_corrupt_unknown_and_offline_reuse(tmp_path, pinned_model, monkeypatch):
    payload, path = pinned_model
    with pytest.raises(T2000Error, match="setup_quality_models"):
        models.verified_model_path(tmp_path, "model.onnx")
    with pytest.raises(T2000Error, match="Unknown"):
        models.verified_model_path(tmp_path, "../other.onnx")
    requests = []

    def urlopen(*args, **kwargs):
        requests.append(args)
        return io.BytesIO(payload)

    monkeypatch.setattr(models.urllib.request, "urlopen", urlopen)
    models.download_quality_models(tmp_path)
    assert models.verified_model_path(tmp_path, "model.onnx") == path
    models.download_quality_models(tmp_path)
    assert len(requests) == 1
    path.write_bytes(b"!" + payload[1:])
    with pytest.raises(T2000Error, match="SHA-256"):
        models.verified_model_path(tmp_path, "model.onnx")


def test_failed_download_never_publishes_partial_or_replaces_previous_file(tmp_path, pinned_model, monkeypatch):
    payload, path = pinned_model
    path.parent.mkdir()
    path.write_bytes(b"old invalid model")
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="mismatch"):
        models.download_quality_models(tmp_path)
    assert path.read_bytes() == b"old invalid model"
    assert list(path.parent.iterdir()) == [path]


def test_reports_keep_failures_and_embed_listening_audio(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/compare_voice_quality.py"
    spec = importlib.util.spec_from_file_location("quality_compare_test", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    wav = tmp_path / "audio.wav"
    sf.write(wav, np.full(22050, 0.1), 22050)
    good = row(script.BASELINE)
    good.update(wav_path=str(wav), reference_path=str(wav))
    report = {"records": [good], "variants": [{"id": script.BASELINE}],
              "references": [{"reference_sha256": "hash1"}]}
    script.write_products(report, tmp_path)
    assert "data:audio/wav;base64," in (tmp_path / "LISTEN.html").read_text()
    assert (tmp_path / "requests.csv").read_text().count("\n") == 2
    assert (tmp_path / "report.json").is_file()


def test_candidate_engine_settings_and_backend_share_requested_conversion_tau(tmp_path, monkeypatch):
    from t2000.config import Settings

    path = Path(__file__).resolve().parents[1] / "scripts/compare_voice_quality.py"
    spec = importlib.util.spec_from_file_location("quality_candidate_test", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    seen = {}

    def backend(*args, **kwargs):
        seen["backend_tau"] = kwargs["tau"]
        return SimpleNamespace(info=lambda: {})

    class EngineBoundary:
        def __init__(self, settings, store, backend):
            seen["settings"] = settings
            self.backend = backend

    monkeypatch.setattr(script, "MeloOpenVoiceBackend", backend)
    monkeypatch.setattr(script, "Engine", EngineBoundary)
    monkeypatch.setattr(script, "release_models", lambda engine: None)
    variant = {"id": "EN-BR/speech/tau=0.6", "speaker": "EN-BR", "reference_mode": "speech", "tau": 0.6}
    script.generate_variant(variant, Settings(), None, [], [], [42], 0, {"setup": []}, tmp_path)
    assert seen["settings"].conversion_tau == seen["backend_tau"] == 0.6
    assert seen["settings"].speaker == "EN-BR" and seen["settings"].reference_mode == "speech"

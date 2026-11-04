"""Sentence pauses keep playback, exported audio, and duration limits consistent."""

from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from test_engine import setup_engine as setup_engine
from t2000.backends.base import AudioChunk
from t2000.errors import T2000Error


def audio_packets(events):
    return [event for event in events if event["type"] == "audio"]


def streamed_samples(events):
    return np.concatenate([np.frombuffer(packet["pcm"], dtype="<f4") for packet in audio_packets(events)])


def test_sentence_pauses_have_the_same_order_in_stream_and_export(setup_engine, monkeypatch):
    engine, profile, backend = setup_engine
    speech = [np.full(2400, 0.25, dtype=np.float32),
              np.full(3200, 0.5, dtype=np.float32),
              np.full(1600, -0.25, dtype=np.float32)]
    phrases = {"First.": speech[0], "Second.": speech[1], "Third.": speech[2]}
    monkeypatch.setattr(backend, "synthesize", lambda text, *args, **kwargs: AudioChunk(phrases[text], 16000))
    events = []

    result = engine.generate(profile["id"], "First. Second. Third.", sentence_gap=0.25, emit=events.append)

    pause = np.zeros(4000, dtype=np.float32)
    expected = np.concatenate([speech[0], pause, speech[1], pause, speech[2]])
    packets = audio_packets(events)
    assert [packet["sequence"] for packet in packets] == list(range(len(packets)))
    assert all(0 < packet["frames"] <= 1600 for packet in packets)
    np.testing.assert_array_equal(streamed_samples(events), expected)
    saved, rate = sf.read(engine.settings.data_dir / "output" / f"{result['job_id']}.wav")
    assert rate == 16000
    np.testing.assert_array_equal(saved, expected)
    assert result["metrics"]["spoken_seconds"] == pytest.approx(0.45)
    assert result["metrics"]["inserted_silence_seconds"] == pytest.approx(0.5)
    assert result["metrics"]["audio_seconds"] == pytest.approx(0.95)
    assert result["metrics"]["rtf"] == pytest.approx(result["metrics"]["generation_seconds"] / 0.45)


def test_zero_pause_preserves_default_wav_bytes_and_stream_samples(setup_engine):
    engine, profile, _ = setup_engine
    default_events, zero_events = [], []
    default = engine.generate(profile["id"], "First. Second.", emit=default_events.append)
    zero = engine.generate(profile["id"], "First. Second.", sentence_gap=0, emit=zero_events.append)
    output = engine.settings.data_dir / "output"

    assert (output / f"{default['job_id']}.wav").read_bytes() == (output / f"{zero['job_id']}.wav").read_bytes()
    np.testing.assert_array_equal(streamed_samples(default_events), streamed_samples(zero_events))
    assert zero["metrics"]["audio_seconds"] == pytest.approx(0.6)
    assert zero["metrics"]["spoken_seconds"] == pytest.approx(0.6)
    assert zero["metrics"]["inserted_silence_seconds"] == 0


@pytest.mark.parametrize("gap", [-0.1, 2.1, float("nan"), float("inf"), True, "0.5", None])
def test_invalid_sentence_pause_is_rejected_before_synthesis(setup_engine, gap):
    engine, profile, backend = setup_engine
    with pytest.raises(T2000Error, match="Sentence pause"):
        engine.generate(profile["id"], "First. Second.", sentence_gap=gap)
    assert backend.prepared == []
    assert backend.texts == []
    assert not (engine.settings.data_dir / "output").exists()


def test_output_limit_counts_inserted_silence_and_removes_partial_output(setup_engine):
    engine, profile, _ = setup_engine
    engine.settings = replace(engine.settings, max_output_seconds=0.8)
    # Two 0.3 s phrases fit by themselves, but a 0.25 s pause crosses the limit.
    with pytest.raises(T2000Error, match="shorter sentence pause"):
        engine.generate(profile["id"], "First. Second.", sentence_gap=0.25)

    assert list((engine.settings.data_dir / "output").iterdir()) == []
    # The failed job releases the lock and leaves generation usable.
    assert engine.generate(profile["id"], "First. Second.", sentence_gap=0)["type"] == "complete"

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf

from t2000.backends.base import AudioChunk
from t2000.config import Settings
from t2000.engine import Cancelled, Engine
from t2000.errors import T2000Error
from t2000.profiles import ProfileStore
from t2000.text import split_phrases


class FakeBackend:
    """A deterministic signal source only for transport tests, never an app backend."""
    def __init__(self):
        self.fail = False
        self.active = 0
        self.max_active = 0
        self.prepared = []

    def load(self):
        if self.fail:
            raise T2000Error("Missing test weights")

    def signature(self):
        return "test-backend-v1"

    def prepare(self, reference_path, cache_dir):
        self.prepared.append(cache_dir)

    def synthesize(self, text, cache_dir, *, speed, seed):
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        time.sleep(0.005)
        self.active -= 1
        return AudioChunk((0.2 * np.sin(np.arange(4800) * 0.1)).astype(np.float32), 16000,
                          base_seconds=0.002, conversion_seconds=0.003)

    def info(self):
        return {"backend": "test-double"}


@pytest.fixture
def setup_engine(tmp_path):
    reference = tmp_path / "reference.wav"
    sf.write(reference, 0.1 * np.sin(np.arange(48000) * 0.03), 48000)
    store = ProfileStore(tmp_path / "profiles")
    profile = store.create(reference, "Voice")
    backend = FakeBackend()
    settings = Settings(data_dir=tmp_path)
    return Engine(settings, store, backend), profile, backend


def test_complete_output_matches_stream_and_is_published_atomically(setup_engine):
    engine, profile, _ = setup_engine
    events = []
    result = engine.generate(profile["id"], "Hello there! This is a second phrase.", emit=events.append)
    assert result["type"] == "complete"
    packets = [e for e in events if e["type"] == "audio"]
    assert [e["sequence"] for e in packets] == list(range(len(packets)))
    samples = np.concatenate([np.frombuffer(e["pcm"], dtype="<f4") for e in packets])
    saved, rate = sf.read(engine.settings.data_dir / "output" / f"{result['job_id']}.wav")
    np.testing.assert_allclose(saved, samples, atol=1 / 32768)
    assert rate == 16000
    assert result["metrics"]["audio_seconds"] == pytest.approx(len(samples) / rate)
    assert not list((engine.settings.data_dir / "output").glob("*.part"))


def test_cancellation_does_not_publish_a_partial_file(setup_engine):
    engine, profile, _ = setup_engine
    cancel = threading.Event()
    def emit(event):
        if event["type"] == "audio":
            cancel.set()
    with pytest.raises(Cancelled):
        engine.generate(profile["id"], "Some words.", emit=emit, cancel=cancel)
    assert list((engine.settings.data_dir / "output").iterdir()) == []
    assert not engine.status()["busy"]


def test_load_error_releases_worker_for_next_request(setup_engine):
    engine, profile, backend = setup_engine
    backend.fail = True
    with pytest.raises(T2000Error, match="Missing"):
        engine.generate(profile["id"], "Hello.")
    backend.fail = False
    assert engine.generate(profile["id"], "Hello.")["type"] == "complete"


def test_model_calls_are_serialized(setup_engine):
    engine, profile, backend = setup_engine
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: engine.generate(profile["id"], "Hello."), range(2)))
    assert len(results) == 2
    assert backend.max_active == 1


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.zeros(100), np.zeros((2, 100))])
def test_invalid_model_audio_is_rejected(setup_engine, bad):
    engine, profile, backend = setup_engine
    backend.synthesize = lambda *args, **kwargs: AudioChunk(bad, 16000)
    with pytest.raises(T2000Error):
        engine.generate(profile["id"], "Hello.")
    assert list((engine.settings.data_dir / "output").iterdir()) == []


def test_phrase_split_preserves_words_and_bounds_length():
    text = "The total is 12.50 dollars. Please come tomorrow! " + "word " * 100
    phrases = split_phrases(text, 80)
    assert " ".join(phrases).split() == text.split()
    assert all(len(phrase) <= 80 for phrase in phrases)
    assert "12.50" in phrases[0]


@pytest.mark.parametrize("options", [{"speed": float("nan")}, {"seed": True}, {"seed": -1}])
def test_invalid_generation_settings(setup_engine, options):
    engine, profile, _ = setup_engine
    with pytest.raises(T2000Error):
        engine.generate(profile["id"], "Hello.", **options)

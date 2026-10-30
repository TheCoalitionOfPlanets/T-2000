"""Regression checks for resource cleanup and malformed transport messages."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from t2000.api import create_app
from t2000.config import Settings
from t2000.engine import Engine
from t2000.errors import T2000Error
from t2000.profiles import ProfileStore
from test_engine import FakeBackend


@pytest.fixture
def edge_engine(tmp_path):
    source = tmp_path / "reference.wav"
    sf.write(source, 0.1 * np.sin(np.arange(48000) * 0.03), 48000)
    store = ProfileStore(tmp_path / "profiles")
    profile = store.create(source, "Reference")
    backend = FakeBackend()
    engine = Engine(Settings(data_dir=tmp_path), store, backend)
    return engine, profile, backend


def test_output_directory_setup_failure_releases_engine_lock(edge_engine):
    engine, profile, _ = edge_engine
    (engine.settings.data_dir / "output").write_text("a file blocks directory creation")
    with pytest.raises(OSError):
        engine.generate(profile["id"], "Hello.")
    assert not engine._lock.locked()


def test_cleanup_failure_preserves_original_error_and_releases_lock(edge_engine, monkeypatch):
    engine, profile, backend = edge_engine
    backend.fail = True
    original = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if str(path).endswith(".part"):
            raise PermissionError("Simulated cleanup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(T2000Error, match="Missing test weights"):
        engine.generate(profile["id"], "Hello.")
    assert not engine._lock.locked()


def test_report_publication_failure_does_not_publish_wav(edge_engine, monkeypatch):
    engine, profile, _ = edge_engine
    original = Path.replace

    def fail_report(path, target):
        if str(path).endswith(".json.part"):
            raise PermissionError("Simulated report publication failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_report)
    with pytest.raises(OSError, match="report publication"):
        engine.generate(profile["id"], "Hello.")
    output = engine.settings.data_dir / "output"
    assert not list(output.glob("*.wav"))
    assert not engine._lock.locked()


def test_binary_initial_request_is_handled_as_protocol_error(tmp_path):
    app = create_app(Settings(data_dir=tmp_path), FakeBackend())
    with TestClient(app) as client:
        with client.websocket_connect("/api/generate") as socket:
            socket.send_bytes(b"unexpected binary request")
            event = socket.receive_json()
            assert event["type"] == "error"

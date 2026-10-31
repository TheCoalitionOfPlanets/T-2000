import io
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from t2000.api import create_app
from t2000.config import Settings
from test_engine import FakeBackend


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, model_dir=tmp_path / "models"), FakeBackend())
    with TestClient(app) as client:
        yield client


def reference_bytes():
    out = io.BytesIO()
    sf.write(out, 0.1 * np.sin(np.arange(48000) * 0.03), 48000, format="WAV")
    return out.getvalue()


def enroll(client):
    response = client.post("/api/profiles", files={"file": ("reference.wav", reference_bytes(), "audio/wav")},
                           data={"name": "My voice"})
    assert response.status_code == 201, response.text
    return response.json()


def test_enrollment_keeps_native_rate_and_editable_transcript(client):
    profile = enroll(client)
    assert profile["sample_rate"] == 48000
    assert client.get("/api/profiles").json()[0]["id"] == profile["id"]
    updated = client.patch(f"/api/profiles/{profile['id']}", json={"transcript": "Hello there"})
    assert updated.json()["transcript"] == "Hello there"
    assert client.get(f"/api/profiles/{profile['id']}/audio").status_code == 200


def test_binary_audio_metadata_and_completed_download(client):
    profile = enroll(client)
    with client.websocket_connect("/api/generate") as socket:
        socket.send_json({"profile_id": profile["id"], "text": "Hello! Second phrase."})
        packets = 0
        while True:
            event = socket.receive_json()
            assert event["type"] != "error", event
            if event["type"] == "audio":
                pcm = socket.receive_bytes()
                assert len(pcm) == event["frames"] * 4
                assert event["sequence"] == packets
                packets += 1
            elif event["type"] == "complete":
                assert packets > 0
                assert client.get(event["audio_url"]).status_code == 200
                break


def test_invalid_upload_does_not_create_profile(client):
    response = client.post("/api/profiles", files={"file": ("bad.wav", b"bad audio")})
    assert response.status_code == 400
    assert client.get("/api/profiles").json() == []


def test_cross_origin_writes_are_rejected(client):
    response = client.post("/api/profiles", headers={"Origin": "https://elsewhere.example"})
    assert response.status_code == 403


def test_optional_stt_failure_preserves_reference(client):
    response = client.post("/api/profiles", files={"file": ("reference.wav", reference_bytes())},
                           data={"name": "Voice", "transcribe": "true"})
    assert response.status_code == 201
    assert response.json()["transcription_error"]
    assert len(client.get("/api/profiles").json()) == 1


def test_stt_is_default_and_existing_corrections_survive_reenrollment(client, monkeypatch):
    calls = []

    def transcribe(path, **kwargs):
        calls.append(path)
        return "Recognized reference words"

    monkeypatch.setattr("t2000.stt.transcribe_file", transcribe)
    profile = enroll(client)
    assert profile["transcript"] == "Recognized reference words"
    assert len(calls) == 1
    client.patch(f"/api/profiles/{profile['id']}", json={"transcript": "My corrected words"})
    assert enroll(client)["transcript"] == "My corrected words"
    assert len(calls) == 1


@pytest.mark.parametrize("data", [{"transcribe": "false"}, {"transcript": "Manually checked words"}])
def test_stt_optout_and_manual_transcripts_do_not_recognize(client, monkeypatch, data):
    def forbidden(*args, **kwargs):
        pytest.fail("Recognition should not overwrite or ignore the user's choice")

    monkeypatch.setattr("t2000.stt.transcribe_file", forbidden)
    response = client.post("/api/profiles", files={"file": ("reference.wav", reference_bytes())}, data=data)
    assert response.status_code == 201
    assert response.json()["transcript"] == data.get("transcript", "")


def test_explicit_transcript_corrects_deduplicated_profile(client):
    profile = enroll(client)
    response = client.post("/api/profiles", files={"file": ("reference.wav", reference_bytes())},
                           data={"transcript": "Checked replacement"})
    assert response.json()["id"] == profile["id"]
    assert response.json()["transcript"] == "Checked replacement"


def test_manual_correction_during_recognition_wins(client, monkeypatch):
    def transcribe(path, **kwargs):
        client.app.state.store.update_transcript(path.parent.name, "Correction made during recognition")
        return "Late automatic result"

    monkeypatch.setattr("t2000.stt.transcribe_file", transcribe)
    profile = enroll(client)
    assert profile["transcript"] == "Correction made during recognition"
    assert client.get("/api/profiles").json()[0]["transcript"] == "Correction made during recognition"


def test_direct_clone_rejects_long_reference_before_creating_profile(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, backend="f5-tts-v1"), FakeBackend())
    reference = io.BytesIO()
    sf.write(reference, 0.1 * np.sin(np.arange(16000 * 13) * 0.04), 16000, format="WAV")
    with TestClient(app) as direct_client:
        response = direct_client.post("/api/profiles", files={"file": ("long.wav", reference.getvalue())})
        assert response.status_code == 400
        assert "12 seconds" in response.json()["detail"]
        assert direct_client.get("/api/profiles").json() == []


def test_invalid_generation_rejected_before_work(client):
    with client.websocket_connect("/api/generate") as socket:
        socket.send_json({"profile_id": "../reference", "text": "Hello."})
        assert socket.receive_json()["type"] == "error"


def test_output_path_cannot_escape_directory(client):
    assert client.get("/api/output/not-a-job.wav").status_code == 404


def test_backend_switch_releases_old_worker_and_preserves_voice(client, monkeypatch):
    profile = enroll(client)
    engine = client.app.state.engine
    old = engine.backend
    closed = []
    old.close = lambda: closed.append(True)
    monkeypatch.setattr(engine, "available_backends", lambda: [
        {"id": "t1000-soar", "available": True, "label": "Reference quality"},
    ])
    response = client.post("/api/backend", json={"backend": "t1000-soar"})
    assert response.status_code == 200
    assert closed == [True]
    assert engine._backend is None
    assert response.json()["backend"] == "t1000-soar"
    assert response.json()["max_reference_seconds"] == 6
    assert response.json()["busy"] is False
    assert client.get("/api/profiles").json()[0]["id"] == profile["id"]


def test_backend_switch_rejects_unavailable_or_busy_engine(client, monkeypatch):
    engine = client.app.state.engine
    monkeypatch.setattr(engine, "available_backends", lambda: [{"id": "t1000-soar", "available": False}])
    assert client.post("/api/backend", json={"backend": "t1000-soar"}).status_code == 400
    monkeypatch.setattr(engine, "available_backends", lambda: [{"id": "t1000-soar", "available": True}])
    engine._lock.acquire()
    try:
        response = client.post("/api/backend", json={"backend": "t1000-soar"})
        assert response.status_code == 400
        assert engine.settings.backend == "melo-openvoice-v2"
    finally:
        engine._lock.release()
    assert client.post("/api/backend", json={"backend": "not-an-engine"}).status_code == 422


def crop_reference_bytes():
    rate = 48000
    samples = np.zeros(rate * 4, dtype=np.float32)
    samples[rate:rate * 3] = 0.2 * np.sin(2 * np.pi * 220 * np.arange(rate * 2) / rate)
    out = io.BytesIO()
    sf.write(out, samples, rate, format="WAV", subtype="FLOAT")
    return out.getvalue()


def test_preview_and_saved_reference_are_identical_and_original_stays_immutable(client):
    source = crop_reference_bytes()
    original = client.post("/api/profiles", files={"file": ("original.wav", source)},
                           data={"name": "Original", "noise_reduction": "false"}).json()
    original_audio = client.get(f"/api/profiles/{original['id']}/audio").content
    options = {"start_seconds": "1", "end_seconds": "3", "noise_reduction": "true"}
    preview = client.post("/api/reference/preview", files={"file": ("source.wav", source)}, data=options)
    assert preview.status_code == 200, preview.text
    assert preview.headers["cache-control"] == "no-store"
    assert float(preview.headers["x-reference-duration"]) == 2
    assert len(client.get("/api/profiles").json()) == 1, "preview must not save a profile"
    saved = client.post("/api/profiles", files={"file": ("source.wav", source)},
                        data=dict(options, name="Edited")).json()
    assert saved["id"] != original["id"]
    assert saved["sample_rate"] == 48000
    assert saved["duration_seconds"] == 2
    assert saved["analysis"]["duration_seconds"] == 2
    assert saved["reference_processing"]["start_seconds"] == 1
    saved_audio = client.get(f"/api/profiles/{saved['id']}/audio").content
    assert saved_audio == preview.content
    assert client.get(f"/api/profiles/{original['id']}/audio").content == original_audio
    assert list((client.app.state.engine.settings.data_dir / ".uploads").iterdir()) == []


@pytest.mark.parametrize("options", [
    {"start_seconds": "3", "end_seconds": "2"}, {"start_seconds": "-1"},
    {"end_seconds": "5"}, {"start_seconds": "nan"}, {"end_seconds": "inf"},
    {"start_seconds": "1", "end_seconds": "1.5"}, {"start_seconds": "0", "end_seconds": "1"},
])
def test_invalid_crop_rejected_by_preview_and_save_without_artifacts(client, options):
    for endpoint in ("/api/reference/preview", "/api/profiles"):
        response = client.post(endpoint, files={"file": ("source.wav", crop_reference_bytes())}, data=options)
        assert response.status_code in (400, 422), response.text
    assert client.get("/api/profiles").json() == []
    assert list((client.app.state.engine.settings.data_dir / ".uploads").iterdir()) == []


def test_transcription_receives_the_processed_crop(client):
    captured = []

    def recognize(path, **kwargs):
        samples, rate = sf.read(path)
        captured.append((len(samples), rate))
        return "The kept words."

    with patch("t2000.stt.transcribe_file", side_effect=recognize):
        response = client.post("/api/profiles", files={"file": ("source.wav", crop_reference_bytes())},
                               data={"start_seconds": "1", "end_seconds": "3", "transcribe": "true"})
    assert response.status_code == 201, response.text
    assert response.json()["transcript"] == "The kept words."
    assert captured == [(96000, 48000)]


def test_noise_cleanup_is_default_for_api_enrollment(client):
    profile = enroll(client)
    assert profile["reference_processing"]["noise_reduction"] is True
    assert profile["preprocessing_version"].endswith("cleanup-v1")


@pytest.mark.parametrize("backend,limit", [("f5-tts-v1", 12), ("t1000-soar", 6)])
def test_backend_limit_applies_to_kept_audio_and_cropped_transcripts_can_be_corrected(
    tmp_path, monkeypatch, backend, limit,
):
    rate = 16000
    samples = (0.1 * np.sin(np.arange(rate * 13) * 0.04)).astype(np.float32)
    source = io.BytesIO()
    sf.write(source, samples, rate, format="WAV", subtype="FLOAT")
    captured = []

    def transcribe(path, **kwargs):
        kept, kept_rate = sf.read(path, dtype="float32")
        assert kept_rate == rate
        np.testing.assert_array_equal(kept, samples[rate:rate * (limit + 1)])
        captured.append(path)
        return "Automatically recognized kept words"

    monkeypatch.setattr("t2000.stt.transcribe_file", transcribe)
    app = create_app(Settings(data_dir=tmp_path, backend=backend), FakeBackend())
    options = {"start_seconds": "1", "end_seconds": str(limit + 1), "noise_reduction": "false"}
    with TestClient(app) as direct_client:
        response = direct_client.post("/api/profiles", files={"file": ("long.wav", source.getvalue())},
                                      data=options)
        assert response.status_code == 201, response.text
        profile = response.json()
        assert profile["duration_seconds"] == limit
        assert profile["transcript"] == "Automatically recognized kept words"
        corrected = direct_client.post("/api/profiles", files={"file": ("long.wav", source.getvalue())},
                                       data=dict(options, transcript="Manually corrected kept words"))
        assert corrected.status_code == 201, corrected.text
        assert corrected.json()["id"] == profile["id"]
        assert corrected.json()["transcript"] == "Manually corrected kept words"
        assert len(captured) == 1
        assert len(direct_client.get("/api/profiles").json()) == 1
        assert list((tmp_path / ".uploads").iterdir()) == []


@pytest.mark.parametrize("backend,limit", [("f5-tts-v1", 12), ("t1000-soar", 6)])
def test_over_limit_crop_can_be_previewed_but_cannot_create_profile(tmp_path, monkeypatch, backend, limit):
    source = io.BytesIO()
    sf.write(source, 0.1 * np.sin(np.arange(16000 * 13) * 0.04), 16000, format="WAV")

    def forbidden(*args, **kwargs):
        pytest.fail("An over-limit crop must be rejected before transcription")

    monkeypatch.setattr("t2000.stt.transcribe_file", forbidden)
    app = create_app(Settings(data_dir=tmp_path, backend=backend), FakeBackend())
    options = {"end_seconds": str(limit + 0.5), "noise_reduction": "false"}
    with TestClient(app) as direct_client:
        preview = direct_client.post("/api/reference/preview", files={"file": ("long.wav", source.getvalue())},
                                     data=options)
        assert preview.status_code == 200, preview.text
        assert float(preview.headers["x-reference-duration"]) == limit + 0.5
        response = direct_client.post("/api/profiles", files={"file": ("long.wav", source.getvalue())},
                                      data=options)
        assert response.status_code == 400, response.text
        assert f"{limit} seconds" in response.json()["detail"]
        assert direct_client.get("/api/profiles").json() == []
        assert list((tmp_path / "profiles").iterdir()) == []
        assert list((tmp_path / ".uploads").iterdir()) == []

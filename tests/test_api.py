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


def test_invalid_generation_rejected_before_work(client):
    with client.websocket_connect("/api/generate") as socket:
        socket.send_json({"profile_id": "../reference", "text": "Hello."})
        assert socket.receive_json()["type"] == "error"


def test_output_path_cannot_escape_directory(client):
    assert client.get("/api/output/not-a-job.wav").status_code == 404


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

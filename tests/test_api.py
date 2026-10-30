import io

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

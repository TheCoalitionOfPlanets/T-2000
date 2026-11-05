import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from t2000.config import Settings
from t2000.errors import T2000Error
from t2000.profiles import ProfileStore
from t2000.solo import create_solo_app, resolve_profile
from test_engine import FakeBackend


@pytest.fixture
def store(tmp_path):
    reference = tmp_path / "reference.wav"
    sf.write(reference, 0.1 * np.sin(np.arange(48000) * 0.03), 48000)
    store = ProfileStore(tmp_path / "profiles")
    store.create(reference, "Prieyan", "Hello there")
    return store


@pytest.fixture
def client(tmp_path, store):
    backend = FakeBackend()
    app = create_solo_app(Settings(data_dir=tmp_path, model_dir=tmp_path / "models"),
                          profile="Prieyan", backend=backend)
    with TestClient(app) as client:
        client.backend = backend
        yield client


def test_resolves_a_saved_voice_by_name_id_and_prefix(tmp_path, store):
    profile = store.list()[0]
    assert resolve_profile(store, "Prieyan")["id"] == profile["id"]
    assert resolve_profile(store, "prieyan")["id"] == profile["id"]
    assert resolve_profile(store, profile["id"])["id"] == profile["id"]
    assert resolve_profile(store, profile["id"][:12])["id"] == profile["id"]
    with pytest.raises(T2000Error, match="No saved voice matches"):
        resolve_profile(store, "Someone else")


def test_ambiguous_name_asks_for_the_identifier(tmp_path, store):
    second = tmp_path / "second.wav"
    sf.write(second, 0.1 * np.sin(np.arange(48000) * 0.05), 48000)
    store.create(second, "Prieyan")
    with pytest.raises(T2000Error, match="more than one saved voice"):
        resolve_profile(store, "Prieyan")


def test_describes_only_the_pinned_voice(client, store):
    voice = client.get("/api/voice").json()
    assert voice["voice"] == "Prieyan"
    assert voice["title"] == "Prieyan-TTS"
    assert voice["profile_id"] == store.list()[0]["id"]
    assert voice["warm"]["state"] in {"cold", "warming", "ready"}
    assert client.get("/api/reference").status_code == 200
    # The studio's enrollment and voice-switching surface is absent here.
    for path in ("/api/profiles", "/api/status", "/api/backend"):
        assert client.get(path).status_code == 404


def test_speaks_without_a_client_supplied_profile(client):
    with client.websocket_connect("/api/speak") as socket:
        socket.send_json({"text": "Hello! Second phrase.", "speed": 1.0, "seed": 7})
        packets = 0
        while True:
            event = socket.receive_json()
            assert event["type"] != "error", event
            if event["type"] == "audio":
                assert len(socket.receive_bytes()) == event["frames"] * 4
                packets += 1
            elif event["type"] == "complete":
                assert packets > 0
                assert client.get(event["audio_url"]).status_code == 200
                break


def test_rejects_a_request_naming_another_voice(client):
    with client.websocket_connect("/api/speak") as socket:
        socket.send_json({"text": "Hello", "profile_id": "a" * 64})
        event = socket.receive_json()
        assert event["type"] == "error" and "Invalid speech request" in event["message"]


def test_warming_prepares_once_and_repeat_requests_reuse_it(client):
    assert client.post("/api/warm").json()["ready"] is True
    prepared = len(client.backend.prepared)
    assert prepared >= 1
    for _ in range(2):
        with client.websocket_connect("/api/speak") as socket:
            socket.send_json({"text": "Again."})
            while (event := socket.receive_json())["type"] != "complete":
                assert event["type"] != "error", event
                if event["type"] == "audio":
                    socket.receive_bytes()
    # The conditioned reference is unchanged, so the backend is not re-prepared.
    assert len(client.backend.prepared) == prepared
    assert client.get("/api/voice").json()["warm"]["state"] == "ready"


def test_list_shows_every_voice_and_flags_shared_names(tmp_path, store, capsys):
    from t2000.cli import print_voices

    # Profiles are content-addressed, so each voice needs its own distinct audio.
    for name, (path, tone) in {"Prieyan": ("second.wav", 0.05), "Solo name": ("third.wav", 0.07)}.items():
        source = tmp_path / path
        sf.write(source, 0.1 * np.sin(np.arange(48000) * tone), 48000)
        store.create(source, name, "A take")
    print_voices(store)
    out = capsys.readouterr().out
    assert "3 saved voices" in out
    for profile in store.list():
        assert profile["id"][:12] in out
    # A shared name cannot select a voice, so the listing has to say so.
    assert out.count("Prieyan *") == 2
    assert "Solo name" in out and "Solo name *" not in out
    assert "select those by ID" in out


def test_list_of_an_empty_store_points_at_enrollment(tmp_path, capsys):
    from t2000.cli import print_voices

    print_voices(ProfileStore(tmp_path / "empty"))
    out = capsys.readouterr().out
    assert "No saved voices yet" in out and "t2000 enroll" in out

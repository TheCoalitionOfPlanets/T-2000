"""The high-VRAM profile's wiring: sources, limits, worker arguments, downloader."""

import importlib.util
from pathlib import Path
import sys

import pytest

from t2000.backends.t1000 import (PROFILE_REFERENCE_SECONDS, PROFILE_SOURCES, T1000Backend)
from t2000.config import Settings
from t2000.errors import T2000Error


ROOT = Path(__file__).resolve().parents[1]


def backend(tmp_path, profile="high-vram", **kwargs):
    return T1000Backend(tmp_path, python_executable=Path(sys.executable), t1000_root=tmp_path,
                        model="mf2", device="cuda:0", profile=profile, **kwargs)


def test_vendored_runtime_ships_every_source_each_profile_verifies():
    for profile, names in PROFILE_SOURCES.items():
        for name in names:
            assert (ROOT / "runtime/t1000" / name).is_file(), f"{profile} needs runtime/t1000/{name}"


def test_high_vram_profile_raises_the_reference_limit_to_the_models_own_cap(tmp_path):
    assert PROFILE_REFERENCE_SECONDS["low-vram"] == 6
    # dots.tts crops the speaker encoder at config.xvec_max_audio_seconds = 10.
    assert PROFILE_REFERENCE_SECONDS["high-vram"] == 10
    assert backend(tmp_path, "high-vram").max_reference_seconds == 10
    assert backend(tmp_path, "low-vram").max_reference_seconds == 6


def test_unknown_profile_is_rejected(tmp_path):
    with pytest.raises(T2000Error, match="low-vram or high-vram"):
        backend(tmp_path, "turbo")


def test_profile_reaches_the_worker_command_and_identity(tmp_path, monkeypatch):
    source = tmp_path / "runtime"
    model = backend(tmp_path, "high-vram", runtime_source=source)
    assert model.runtime_source == source.resolve()
    captured = {}

    class Popen:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            self.stdin = self.stdout = None

        def poll(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", Popen)
    monkeypatch.setattr(model, "signature", lambda: "test-signature")
    with pytest.raises(T2000Error):
        model.load()
    command = captured["command"]
    assert "--profile" in command and command[command.index("--profile") + 1] == "high-vram"
    assert command[command.index("--runtime-source") + 1] == str(source.resolve())


def test_default_runtime_source_keeps_the_external_checkout_layout(tmp_path):
    assert backend(tmp_path, "low-vram").runtime_source == (tmp_path / "apps/api/src").resolve()


@pytest.mark.parametrize("value,expected", [("high-vram", 10), ("low-vram", 6)])
def test_engine_reference_limit_follows_the_configured_profile(monkeypatch, value, expected):
    from t2000.engine import Engine

    monkeypatch.setenv("T2000_BACKEND", "t1000-mf2")
    monkeypatch.setenv("T2000_T1000_PROFILE", value)
    assert Engine(Settings.from_env(), None).max_reference_seconds == expected


def test_invalid_profile_env_is_actionable(monkeypatch):
    monkeypatch.setenv("T2000_T1000_PROFILE", "enormous")
    with pytest.raises(T2000Error, match="low-vram or high-vram"):
        Settings.from_env()


def worker_module():
    path = ROOT / "scripts/t1000_infer_worker.py"
    spec = importlib.util.spec_from_file_location("t1000_worker_profiles_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_rejects_an_unknown_profile_before_importing_torch(tmp_path):
    with pytest.raises(ValueError, match="low-vram or high-vram"):
        worker_module().Runtime(tmp_path, "mf2", "cuda:0", 4, profile="turbo")


def test_worker_resolves_the_vendored_source_and_model_directory(tmp_path):
    runtime = worker_module().Runtime(tmp_path, "mf2", "cuda:0", 4,
                                      profile="high-vram", runtime_source=ROOT / "runtime")
    assert runtime.runtime_source == (ROOT / "runtime").resolve()
    assert runtime.model_path == (tmp_path / "data/models/dots.tts-mf-2steps").resolve()


def downloader():
    path = ROOT / "scripts/setup_t1000.py"
    spec = importlib.util.spec_from_file_location("setup_t1000_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_downloader_pins_match_the_adapters_pins():
    from t2000.backends.t1000 import MODEL_PINS

    setup = downloader()
    assert set(setup.REPOS) == set(MODEL_PINS)
    for name, pin in MODEL_PINS.items():
        assert setup.REPOS[name]["revision"] == pin["revision"]
        assert setup.REPOS[name]["sha256"] == pin["sha256"]
        assert setup.REPOS[name]["directory"] == pin["directory"]


def test_downloader_reports_incomplete_directories_without_network(tmp_path):
    setup = downloader()
    assert setup.complete(tmp_path / "dots.tts-soar", setup.REPOS["soar"]) is False
    directory = tmp_path / "dots.tts-soar"
    directory.mkdir()
    for name in setup.MODEL_FILES:
        (directory / name).write_bytes(b"placeholder")
    # Every file exists, but the checkpoint hash must still disqualify it.
    assert setup.complete(directory, setup.REPOS["soar"]) is False

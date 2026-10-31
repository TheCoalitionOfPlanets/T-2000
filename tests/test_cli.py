import json
import sys

import numpy as np
import pytest
import soundfile as sf

from t2000.cli import main
from t2000.config import Settings
from t2000.errors import T2000Error


@pytest.mark.parametrize("arguments,expected_calls,expected_text", [
    ([], 1, "Automatic words"),
    (["--no-transcribe"], 0, ""),
    (["--transcript", "Checked words"], 0, "Checked words"),
])
def test_cli_enrollment_transcribes_by_default_and_respects_overrides(
    tmp_path, monkeypatch, capsys, arguments, expected_calls, expected_text,
):
    reference = tmp_path / "reference.wav"
    sf.write(reference, 0.1 * np.sin(np.arange(24000) * 0.05), 24000)
    calls = []

    def transcribe(path, **kwargs):
        calls.append(path)
        return "Automatic words"

    monkeypatch.setattr("t2000.stt.transcribe_file", transcribe)
    monkeypatch.setenv("T2000_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(sys, "argv", ["t2000", "enroll", str(reference), *arguments])
    main()
    assert json.loads(capsys.readouterr().out)["transcript"] == expected_text
    assert len(calls) == expected_calls


def test_cli_stt_failure_keeps_reference(tmp_path, monkeypatch, capsys):
    reference = tmp_path / "reference.wav"
    sf.write(reference, 0.1 * np.sin(np.arange(24000) * 0.05), 24000)

    def unavailable(*args, **kwargs):
        raise T2000Error("Recognition model unavailable")

    monkeypatch.setattr("t2000.stt.transcribe_file", unavailable)
    monkeypatch.setenv("T2000_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(sys, "argv", ["t2000", "enroll", str(reference)])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["transcription_error"] == "Recognition model unavailable"
    assert (tmp_path / "data" / "profiles" / result["id"] / "reference.wav").is_file()


@pytest.mark.parametrize("name,value", [
    ("T2000_REFERENCE_MODE", "unknown"), ("T2000_CONVERSION_TAU", "nan"),
    ("T2000_CONVERSION_TAU", "-0.1"), ("T2000_CONVERSION_TAU", "1.01"),
    ("T2000_BACKEND", "unknown"), ("T2000_F5_STEPS", "0"),
    ("T2000_F5_STEPS", "65"), ("T2000_F5_STEPS", "nan"),
])
def test_invalid_quality_settings_are_actionable(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(T2000Error, match=name):
        Settings.from_env()


def test_isolated_interpreter_path_keeps_virtual_environment_symlink(tmp_path, monkeypatch):
    base = tmp_path / "base-python"
    base.write_text("base interpreter")
    isolated = tmp_path / "isolated" / "bin" / "python"
    isolated.parent.mkdir(parents=True)
    isolated.symlink_to(base)
    monkeypatch.setenv("T2000_F5_PYTHON", str(isolated))
    assert Settings.from_env().f5_python == isolated


def test_t1000_paths_are_explicit_and_keep_the_virtual_environment(tmp_path, monkeypatch):
    interpreter = tmp_path / "venv/bin/python"
    monkeypatch.setenv("T2000_T1000_ROOT", str(tmp_path / "T-1000"))
    monkeypatch.setenv("T2000_T1000_PYTHON", str(interpreter))
    monkeypatch.setenv("T2000_BACKEND", "t1000-soar")
    settings = Settings.from_env()
    assert settings.backend == "t1000-soar"
    assert settings.t1000_python == interpreter
    assert settings.t1000_root == tmp_path / "T-1000"

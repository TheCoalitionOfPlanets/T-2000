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


@pytest.mark.parametrize("backend,limit", [("f5-tts-v1", 12), ("t1000-soar", 6)])
def test_cli_enrollment_limits_the_crop_and_keeps_manual_transcript_corrections(
    tmp_path, monkeypatch, capsys, backend, limit,
):
    rate = 16000
    samples = (0.1 * np.sin(np.arange(rate * 13) * 0.04)).astype(np.float32)
    reference = tmp_path / "reference.wav"
    sf.write(reference, samples, rate, subtype="FLOAT")
    original = reference.read_bytes()
    captured = []

    def transcribe(path, **kwargs):
        kept, kept_rate = sf.read(path, dtype="float32")
        assert kept_rate == rate
        np.testing.assert_array_equal(kept, samples[rate:rate * (limit + 1)])
        captured.append(path)
        return "Automatically recognized kept words"

    monkeypatch.setattr("t2000.stt.transcribe_file", transcribe)
    monkeypatch.setenv("T2000_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("T2000_BACKEND", backend)
    arguments = ["t2000", "enroll", str(reference), "--start-seconds", "1",
                 "--end-seconds", str(limit + 1), "--no-noise-reduction"]
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    profile = json.loads(capsys.readouterr().out)
    assert profile["duration_seconds"] == limit
    assert profile["transcript"] == "Automatically recognized kept words"
    monkeypatch.setattr(sys, "argv", [*arguments, "--transcript", "Manually corrected kept words"])
    main()
    corrected = json.loads(capsys.readouterr().out)
    assert corrected["id"] == profile["id"]
    assert corrected["transcript"] == "Manually corrected kept words"
    assert len(captured) == 1
    assert reference.read_bytes() == original
    assert len(list((tmp_path / "data" / "profiles").iterdir())) == 1


@pytest.mark.parametrize("backend,limit", [("f5-tts-v1", 12), ("t1000-soar", 6)])
def test_cli_rejects_over_limit_processed_crop_without_artifacts(tmp_path, monkeypatch, capsys, backend, limit):
    reference = tmp_path / "reference.wav"
    sf.write(reference, 0.1 * np.sin(np.arange(16000 * 13) * 0.04), 16000)

    def forbidden(*args, **kwargs):
        pytest.fail("An over-limit crop must be rejected before transcription")

    monkeypatch.setattr("t2000.stt.transcribe_file", forbidden)
    monkeypatch.setenv("T2000_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("T2000_BACKEND", backend)
    monkeypatch.setattr(sys, "argv", ["t2000", "enroll", str(reference), "--end-seconds", str(limit + 0.5)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert f"{limit} seconds" in capsys.readouterr().err
    assert list((tmp_path / "data" / "profiles").iterdir()) == []


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

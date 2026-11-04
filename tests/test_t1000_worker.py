"""CPU regressions for the isolated worker's source layout and reference limits."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from t2000.backends.t1000 import PROFILE_REFERENCE_SECONDS, PROFILE_SOURCES


@pytest.fixture
def worker():
    path = Path(__file__).resolve().parents[1] / "scripts/t1000_infer_worker.py"
    spec = importlib.util.spec_from_file_location("t1000_worker_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_profile_contract_matches_the_adapter(worker):
    assert worker.PROFILE_SOURCES == PROFILE_SOURCES
    assert worker.PROFILE_REFERENCE_SECONDS == PROFILE_REFERENCE_SECONDS


@pytest.mark.parametrize("profile,source_directory", [
    ("low-vram", None),
    ("low-vram", "runtime"),
    ("high-vram", "runtime"),
])
def test_load_hashes_the_selected_runtime_sources(worker, tmp_path, monkeypatch, profile, source_directory):
    runtime_source = tmp_path / source_directory if source_directory else None
    runtime = worker.Runtime(tmp_path, "soar", "cuda:0", 4,
                             profile=profile, runtime_source=runtime_source)
    source = runtime.runtime_source / "t1000"
    source.mkdir(parents=True)
    for name in PROFILE_SOURCES[profile]:
        (source / name).write_text(f"# Selected runtime source: {name}\n")
    runtime.model_path.mkdir(parents=True)
    checkpoint = runtime.model_path / "model.safetensors"
    checkpoint.write_bytes(b"CPU test checkpoint")
    (runtime.model_path / "config.json").write_text("{}")
    directory, revision, _ = worker.PINS["soar"]
    worker.PINS["soar"] = (directory, revision, worker.sha256(checkpoint))

    runtime.torch = Mock()
    runtime.torch.cuda.is_available.return_value = True
    runtime.torch.cuda.is_bf16_supported.return_value = True
    runtime.imports = Mock(return_value={})
    runtime.load_runtime = Mock(return_value=SimpleNamespace(
        max_generate_length=500, max_sequence_length=2048, sample_rate=48000, device="cuda:0",
    ))
    runtime.configure_prompt = Mock()
    runtime.configure_cache = Mock()
    installed_runtime = ModuleType("dots_tts.runtime")
    installed_runtime.__file__ = str(checkpoint)
    package = ModuleType("dots_tts")
    package.runtime = installed_runtime
    monkeypatch.setitem(sys.modules, "dots_tts", package)
    monkeypatch.setitem(sys.modules, "dots_tts.runtime", installed_runtime)

    runtime.load()

    assert runtime.loaded
    assert runtime.provenance["t1000_source_sha256"] == {
        name: worker.sha256(source / name) for name in PROFILE_SOURCES[profile]
    }


@pytest.mark.parametrize("profile,duration,accepted", [
    ("low-vram", 6, True),
    ("low-vram", 7, False),
    ("high-vram", 7, True),
    ("high-vram", 10, True),
    ("high-vram", 10.01, False),
    ("high-vram", 0.99, False),
])
def test_prepare_enforces_the_profile_reference_limit(worker, tmp_path, profile, duration, accepted):
    runtime = worker.Runtime(tmp_path, "soar", "cuda:0", 4, profile=profile)
    runtime.loaded = True
    runtime.np, runtime.sf = np, sf
    runtime.runtime = SimpleNamespace(_load_prompt_audio=Mock())
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.full(round(duration * 8000), 0.1, dtype=np.float32), 8000)
    request = {"reference_path": str(reference), "reference_text": "The matching reference words."}

    if accepted:
        assert runtime.prepare(request)["ok"]
        runtime.runtime._load_prompt_audio.assert_called_once_with(str(reference))
    else:
        with pytest.raises(ValueError, match=f"1–{PROFILE_REFERENCE_SECONDS[profile]} second"):
            runtime.prepare(request)
        runtime.runtime._load_prompt_audio.assert_not_called()

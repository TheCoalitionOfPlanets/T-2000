"""Isolated F5 protocol/preparation checks without optional model imports."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import pytest
import soundfile as sf

from t2000.backends.f5 import F5Backend
from t2000.errors import T2000Error


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_protocol_info_and_invalid_input_do_not_import_models():
    path = Path(__file__).resolve().parents[1] / "scripts/f5_infer_worker.py"
    result = subprocess.run([sys.executable, str(path), "--model-dir", "/unprepared/f5"],
                            input='not json\n{"command":"info"}\n{"command":"close"}\n',
                            text=True, capture_output=True, check=True, timeout=10)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]["ok"] is False
    assert rows[1]["loaded"] is False and rows[1]["type"] == "info"
    assert rows[2]["type"] == "closed"
    assert result.stderr == ""


def test_reference_pacing_changes_duration_and_long_text_keeps_all_words():
    worker = script("f5_infer_worker.py")
    reference_frames, transcript = 900, "These are the exact words of the complete recorded reference. "
    text = "This is a longer sentence that must keep every word in its original order. " * 6
    batches = worker.plan_text_batches(reference_frames, transcript, text.strip(), 0.7)
    assert len(batches) > 1
    assert " ".join(batches).split() == text.split()
    assert all(worker.estimated_duration_frames(reference_frames, transcript, batch, 0.7)
               * worker.HOP_LENGTH / worker.SAMPLE_RATE <= worker.MAX_TOTAL_SECONDS for batch in batches)
    short = "We should finish our conversation before the sun sets."
    assert worker.estimated_duration_frames(reference_frames, transcript, short, 0.8) > \
        worker.estimated_duration_frames(reference_frames, transcript, short, 1.2)


def test_reference_edge_preparation_retains_speech_without_changing_original():
    worker = script("f5_infer_worker.py")
    rate = 24000
    original = np.concatenate([np.zeros(rate), np.full(rate, 0.1), np.zeros(rate // 2)]).astype(np.float32)
    before = original.copy()
    start, end = worker.reference_bounds(original, rate)
    assert (start, end) == (rate, 2 * rate)
    np.testing.assert_array_equal(original, before)


def test_internal_chunk_join_preserves_consonants_and_existing_pauses():
    worker = script("f5_infer_worker.py")
    first = np.full(2400, 0.2, dtype=np.float32)
    second = np.full(2400, -0.15, dtype=np.float32)
    joined, inserted = worker.assemble_chunks([first, second], ["First clause,", "next clause."])
    assert inserted == 0.08
    np.testing.assert_array_equal(joined[:len(first)], first)
    np.testing.assert_array_equal(joined[-len(second):], second)
    assert np.count_nonzero(joined) == len(first) + len(second)
    paused_first = np.concatenate([first, np.zeros(4800, dtype=np.float32)])
    joined, inserted = worker.assemble_chunks([paused_first, second], ["First sentence.", "Next sentence."])
    assert inserted == 0
    np.testing.assert_array_equal(joined, np.concatenate([paused_first, second]))


@pytest.mark.parametrize("override", [{"reference_text": ""}, {"steps": 16.0}, {"seed": True},
                                     {"speed": float("nan")}, {"text": "x" * 601}])
def test_worker_rejects_invalid_requests_before_model_loading(tmp_path, override):
    worker = script("f5_infer_worker.py")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"Presence is sufficient for protocol validation; no decoder is imported.")
    request = {"reference_path": str(reference), "reference_text": "Recorded words.", "text": "New words.",
               "output_path": str(tmp_path / "output.wav"), **override}
    with pytest.raises(ValueError):
        worker.validate_request(request)


def test_setup_check_verifies_source_bytes_and_only_changes_inference_exports(tmp_path, monkeypatch):
    setup = script("setup_f5.py")
    archive = tmp_path / setup.SOURCE_FILE
    archive.parent.mkdir(parents=True)
    prefix = f"F5-TTS-{setup.SOURCE_REVISION}/src/f5_tts/"
    original = b"# The pinned model implementation must remain byte-identical.\n"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(prefix + "model/cfm.py", original)
        output.writestr(prefix + "model/__init__.py", b"from .trainer import Trainer\n")
    monkeypatch.setattr(setup, "FILES", {setup.SOURCE_FILE: {
        "size": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }})
    for relative, content in setup.source_members(tmp_path):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    assert setup.check_bundle(tmp_path)["complete"]
    model = tmp_path / "sources/F5-TTS/src/f5_tts/model/cfm.py"
    assert model.read_bytes() == original
    model.write_bytes(b"changed model kernel")
    with pytest.raises(ValueError, match="source integrity"):
        setup.check_bundle(tmp_path)


@pytest.fixture
def isolated_backend(tmp_path, monkeypatch):
    worker = tmp_path / "fake_worker.py"
    worker.write_text('''import json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
for line in sys.stdin:
    request = json.loads(line)
    command = request.get("command", "synthesize")
    with Path(__file__).with_suffix(".calls").open("a") as out:
        out.write(command + "\\n")
    response = {"ok": True, "request_id": request.get("request_id"), "type": command}
    if command == "synthesize":
        if request["text"] == "partial-line":
            sys.stdout.write('{"unfinished":')
            sys.stdout.flush()
            continue
        samples = np.sin(np.arange(4800) * .1) * .2
        rate = 22050 if request["text"] == "wrong-rate" else 24000
        if request["text"] == "clipped-file": samples = np.ones(4800)
        sf.write(request["output_path"], samples, rate)
        response.update(generation_seconds=.2, clipped_fraction=0, chunks=1)
        if request["text"] == "wrong-id": response["request_id"] = "unrelated"
        if request["text"] == "invalid-metrics": response["generation_seconds"] = float("nan")
    print(json.dumps(response), flush=True)
    if command == "close": break
''')
    backend = F5Backend(tmp_path / "models", python_executable=Path(sys.executable))
    backend._worker_script = worker
    monkeypatch.setattr(backend, "_identity", lambda: {"test_boundary": True, "steps": backend.steps})
    reference, cache = tmp_path / "reference.wav", tmp_path / "cache"
    sf.write(reference, np.sin(np.arange(48000) * .07) * .2, 24000)
    yield backend, reference, cache, worker
    backend.close()


def test_backend_reuses_isolated_worker_and_closes_it(isolated_backend):
    backend, reference, cache, worker = isolated_backend
    backend.prepare(reference, cache, reference_text="The exact recorded words.")
    process = backend._process
    first = backend.synthesize("One complete phrase.", cache, speed=1, seed=42)
    second = backend.synthesize("Another phrase.", cache, speed=1, seed=43)
    assert first.sample_rate == second.sample_rate == 24000
    assert first.samples.ndim == 1 and np.isfinite(first.samples).all()
    assert backend._process is process
    assert worker.with_suffix(".calls").read_text().splitlines() == ["load", "synthesize", "synthesize"]
    backend.close()
    assert process.poll() == 0 and not backend.info()["loaded"]


def test_backend_preserves_venv_interpreter_path_and_rejects_missing_transcript(tmp_path):
    interpreter = tmp_path / ".venv-f5/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    backend = F5Backend(tmp_path / "models", python_executable=interpreter)
    assert backend.python_executable == interpreter.absolute()
    assert backend.python_executable != interpreter.resolve()
    with pytest.raises(T2000Error, match="exact reference transcript"):
        backend.prepare(tmp_path / "missing.wav", tmp_path / "cache")
    assert backend._process is None


def test_backend_rejects_changed_reference_and_mismatched_worker_response(isolated_backend):
    backend, reference, cache, _ = isolated_backend
    backend.prepare(reference, cache, reference_text="Recorded words.")
    original = reference.read_bytes()
    sf.write(reference, np.zeros(48000), 24000)
    with pytest.raises(T2000Error, match="reference or F5 signature changed"):
        backend.synthesize("New words.", cache, speed=1, seed=42)
    reference.write_bytes(original)
    with pytest.raises(T2000Error, match="response ID"):
        backend.synthesize("wrong-id", cache, speed=1, seed=42)
    assert not backend.info()["loaded"]


def test_corrupt_reference_cache_is_actionable_without_generation(isolated_backend):
    backend, reference, cache, worker = isolated_backend
    backend.prepare(reference, cache, reference_text="Recorded words.")
    descriptor = cache / "f5-reference.json"
    profile = json.loads(descriptor.read_text())
    profile.pop("reference_text")
    descriptor.write_text(json.dumps(profile))
    with pytest.raises(T2000Error, match="conditioning is unavailable"):
        backend.synthesize("New words.", cache, speed=1, seed=42)
    assert worker.with_suffix(".calls").read_text().splitlines() == ["load"]


@pytest.mark.parametrize("text,message", [("clipped-file", "clipped audio"),
                                          ("wrong-rate", "dimensions or duration"),
                                          ("invalid-metrics", "invalid audio metrics")])
def test_adapter_checks_actual_output_and_metrics_not_only_worker_success(isolated_backend, text, message):
    backend, reference, cache, _ = isolated_backend
    backend.prepare(reference, cache, reference_text="Recorded words.")
    with pytest.raises(T2000Error, match=message):
        backend.synthesize(text, cache, speed=1, seed=42)
    assert not list(cache.glob("f5-output-*"))


def test_partial_json_response_deadline_is_bounded_and_closes_worker(isolated_backend):
    backend, reference, cache, _ = isolated_backend
    backend.prepare(reference, cache, reference_text="Recorded words.")
    process = backend._process
    started = time.monotonic()
    with pytest.raises(T2000Error, match="response deadline"):
        backend._exchange({"text": "partial-line"}, timeout=0.1)
    assert time.monotonic() - started < 3
    assert process.poll() is not None and backend._process is None


@pytest.mark.parametrize("kwargs", [{"device": "cudabad"}, {"device": "cuda:-1"},
                                   {"num_threads": True}, {"num_threads": 0}, {"num_threads": 33}])
def test_invalid_process_settings_rejected_without_starting_worker(tmp_path, kwargs):
    with pytest.raises(T2000Error, match="device|threads"):
        F5Backend(tmp_path, python_executable=Path(sys.executable), **kwargs)


@pytest.mark.parametrize("values", [{"text": ""}, {"text": "x" * 601}, {"speed": True},
                                   {"speed": float("nan")}, {"seed": True}, {"seed": -1}])
def test_adapter_rejects_invalid_phrase_controls_before_loading(tmp_path, values):
    backend = F5Backend(tmp_path, python_executable=Path(sys.executable))
    with pytest.raises(T2000Error):
        backend.synthesize(**{"text": "Phrase", "speed": 1, "seed": 42, **values}, cache_dir=tmp_path)
    assert backend._process is None


@pytest.mark.parametrize("manifest", [[], {"schema": 1, "complete": True},
                                     {"schema": 1, "complete": True, "files": [], "sources": {}}])
def test_malformed_bundle_manifest_is_actionable(tmp_path, manifest):
    bundle = tmp_path / "f5"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    backend = F5Backend(tmp_path, python_executable=Path(sys.executable))
    with pytest.raises(T2000Error, match="model manifest"):
        backend.signature()


def test_worker_start_failure_releases_temporary_stderr(tmp_path, monkeypatch):
    backend = F5Backend(tmp_path, python_executable=Path(sys.executable))
    monkeypatch.setattr(backend, "signature", lambda: "test-signature")

    def failed_start(*args, **kwargs):
        raise OSError("simulated process failure")

    monkeypatch.setattr(subprocess, "Popen", failed_start)
    with pytest.raises(T2000Error, match="Could not start"):
        backend.load()
    assert backend._stderr is None and backend._process is None


def test_close_does_not_mask_the_real_error_with_a_broken_pipe(tmp_path, monkeypatch):
    """A worker killed between write() and flush() (for example by the OOM killer)
    leaves its stdin broken. close() used to re-raise EPIPE while tidying up that
    stream, which replaced the caller's actual, diagnosable error with a bare
    "[Errno 32] Broken pipe" and no indication of what actually happened.
    """
    backend = F5Backend(tmp_path, python_executable=Path(sys.executable))
    monkeypatch.setattr(backend, "signature", lambda: "test-signature")

    class DeadStdin:
        def write(self, data):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

        def close(self):
            # Flushing on close surfaces the same broken pipe a second time.
            raise BrokenPipeError(32, "Broken pipe")

    class DeadProcess:
        stdin = DeadStdin()
        stdout = DeadStdin()

        def poll(self):
            return None  # Not yet reaped, matching the real race.

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return -9

    backend._process = DeadProcess()
    backend._stderr = tempfile.TemporaryFile()
    backend._stderr.write(b"worker died loading the model\n")
    # close() must swallow the dead pipe, not raise through it.
    backend.close()
    assert backend._process is None

    # The same scenario reached through _exchange(), where the masking bug
    # actually surfaced: the real error must come through, not a bare EPIPE.
    backend._process = DeadProcess()
    backend._stderr = tempfile.TemporaryFile()
    backend._stderr.write(b"worker died loading the model\n")
    with pytest.raises(T2000Error, match="worker output: worker died loading the model") as excinfo:
        backend._exchange({"command": "load"})
    assert "F5 generation failed" in str(excinfo.value)

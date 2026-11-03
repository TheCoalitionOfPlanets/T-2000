"""Bounded local protocol/adapter tests; never load the existing GPU models."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import soundfile as sf

from t2000.backends.t1000 import T1000Backend, condition_reference_text
from t2000.errors import T2000Error


def worker_module():
    path = Path(__file__).resolve().parents[1] / "scripts/t1000_infer_worker.py"
    spec = importlib.util.spec_from_file_location("t1000_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_info_invalid_json_and_close_do_not_import_model_dependencies():
    path = Path(__file__).resolve().parents[1] / "scripts/t1000_infer_worker.py"
    result = subprocess.run([sys.executable, str(path), "--t1000-root", "/missing/t1000"],
                            input='bad json\n{"command":"info"}\n{"command":"close"}\n',
                            text=True, capture_output=True, timeout=5, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]["ok"] is False and rows[1]["loaded"] is False and rows[2]["type"] == "closed"
    assert result.stderr == ""


@pytest.mark.parametrize("values", [{"reference_text": ""}, {"seed": True}, {"seed": -1},
                                   {"text": "x" * 601}, {"output_path": "output.mp3"}])
def test_invalid_worker_requests_fail_before_load(tmp_path, values):
    path = tmp_path / "reference.wav"
    path.write_bytes(b"Only existence is checked before model import")
    request = {"reference_path": str(path), "reference_text": "Recorded words", "text": "New words",
               "output_path": str(tmp_path / "output.wav"), **values}
    with pytest.raises(ValueError):
        worker_module().validate_request(request)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    path = tmp_path / "fake_t1000.py"
    path.write_text('''import json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
for line in sys.stdin:
    r=json.loads(line)
    command=r.get("command","synthesize")
    with Path(__file__).with_suffix(".calls").open("a") as out: out.write(command+"\\n")
    response={"ok":True,"request_id":r.get("request_id"),"type":command}
    if command=="synthesize":
        if r["text"]=="partial":
            sys.stdout.write('{"unfinished":');sys.stdout.flush();continue
        rate=24000 if r["text"]=="wrong-rate" else 48000
        samples=np.ones(4800) if r["text"]=="clipped" else np.sin(np.arange(4800)*.1)*.2
        sf.write(r["output_path"],samples,rate)
        response.update(generation_seconds=.2,stop_reason="eos",chunks=2,output_gain=1)
        if r["text"]=="capped": response["stop_reason"]="schedule_exhausted"
        if r["text"]=="wrong-id": response["request_id"]="other"
    print(json.dumps(response),flush=True)
    if command=="close": break
''')
    model = T1000Backend(tmp_path / "models", python_executable=Path(sys.executable), t1000_root=tmp_path)
    model._worker_script = path
    monkeypatch.setattr(model, "_identity", lambda: {"model": model.model})
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.sin(np.arange(96000) * .04) * .2, 48000, subtype="FLOAT")
    yield model, reference, tmp_path / "cache", path
    model.close()


def test_reuses_worker_preserves_reference_and_exports_native_rate(backend):
    model, reference, cache, worker = backend
    original = reference.read_bytes()
    model.prepare(reference, cache, reference_text="The exact reference words.")
    process = model._process
    model.prepare(reference, cache, reference_text="The exact reference words.")
    first = model.synthesize("A natural sentence.", cache, speed=1, seed=42)
    second = model.synthesize("Another sentence.", cache, speed=1, seed=43)
    assert first.sample_rate == second.sample_rate == 48000
    assert model.info()["backend"] == "t1000-soar" and model._process is process
    assert reference.read_bytes() == original
    assert worker.with_suffix(".calls").read_text().splitlines() == ["load", "prepare", "synthesize", "synthesize"]
    model.close()
    assert process.poll() == 0 and not model.info()["loaded"]


@pytest.mark.parametrize("text,message", [("capped", "generation limit"), ("clipped", "clipped audio"),
                                          ("wrong-rate", "dimensions"), ("wrong-id", "response ID")])
def test_invalid_or_incomplete_output_is_never_accepted(backend, text, message):
    model, reference, cache, _ = backend
    model.prepare(reference, cache, reference_text="Exact words.")
    with pytest.raises(T2000Error, match=message):
        model.synthesize(text, cache, speed=1, seed=42)
    assert not list(cache.glob("t1000-output-*"))


def test_partial_protocol_response_is_deadline_bounded(backend):
    model, reference, cache, _ = backend
    model.prepare(reference, cache, reference_text="Exact words.")
    started = time.monotonic()
    with pytest.raises(T2000Error, match="T-1000.*response deadline"):
        model._exchange({"text": "partial"}, timeout=.1)
    assert time.monotonic() - started < 3 and model._process is None


def test_changed_reference_and_missing_transcript_are_actionable(backend):
    model, reference, cache, _ = backend
    with pytest.raises(T2000Error, match="exact reference transcript"):
        model.prepare(reference, cache)
    assert model._process is None
    model.prepare(reference, cache, reference_text="Exact words.")
    sf.write(reference, np.zeros(96000), 48000)
    with pytest.raises(T2000Error, match="reference preparation"):
        model.synthesize("A sentence.", cache, speed=1, seed=42)


def test_overlong_reference_rejected_without_truncation_or_model_load(backend):
    model, reference, cache, _ = backend
    sf.write(reference, np.full(7 * 48000, .1), 48000)
    with pytest.raises(T2000Error, match="1–6 second"):
        model.prepare(reference, cache, reference_text="Complete seven second reference.")
    assert model._process is None


@pytest.mark.parametrize("values", [{"speed": 1.1}, {"speed": float("nan")}, {"seed": True}, {"text": ""}])
def test_native_delivery_controls_validated_before_loading(tmp_path, values):
    model = T1000Backend(tmp_path, python_executable=Path(sys.executable), t1000_root=tmp_path)
    with pytest.raises(T2000Error):
        model.synthesize(cache_dir=tmp_path, **{"text": "A sentence", "speed": 1, "seed": 42, **values})
    assert model._process is None


def test_interpreter_symlink_preserves_existing_environment(tmp_path):
    interpreter = tmp_path / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    model = T1000Backend(tmp_path, python_executable=interpreter, t1000_root=tmp_path, model="mf2")
    assert model.python_executable == interpreter.absolute() and model.python_executable != interpreter.resolve()
    assert model.info()["backend"] == "t1000-mf2"


@pytest.mark.parametrize("source,expected", [
    ("Hello, this is Dijo", "Hello, this is Dijo."),
    ("Hello, this is Dijo.", "Hello, this is Dijo."),
    ("Are you serious?", "Are you serious?"), ("Wait!", "Wait!"), ("Let me think…", "Let me think…"),
    ('He said "hello"', 'He said "hello."'), ('He said "hello!"', 'He said "hello!"'),
    ("He said ‘hello’", "He said ‘hello.’"), ("He said ‘hello?’", "He said ‘hello?’"),
])
def test_sentence_reference_boundary_preserves_words_quotes_and_existing_endings(source, expected):
    assert condition_reference_text(source, "sentence") == expected
    assert condition_reference_text(source, "raw") == source
    with pytest.raises(T2000Error, match="sentence or raw"):
        condition_reference_text(source, "unknown")


def test_reference_boundary_cache_records_source_without_editing_recording(backend):
    model, reference, cache, _ = backend
    original = reference.read_bytes()
    source = "Hello, this is Dijo"
    model.prepare(reference, cache, reference_text=source)
    saved = json.loads((cache / "t1000-reference.json").read_text())
    assert saved["source_transcript"] == source
    assert saved["conditioning_text"] == saved["reference_text"] == source + "."
    assert saved["reference_boundary"] == "sentence" and saved["reference_boundary_policy"]
    assert reference.read_bytes() == original
    model.synthesize("A natural phrase.", cache, speed=1, seed=42)
    provenance = model.info()["last_worker"]["reference_conditioning"]
    assert provenance["source_transcript"] == source and provenance["conditioning_text"] == source + "."


def test_raw_and_sentence_policy_have_distinct_conditioning_identity(tmp_path, monkeypatch):
    import t2000.backends.t1000 as adapter
    from t2000.quality_models import sha256

    root = tmp_path / "t1000"
    model_path = root / "data/models/dots.tts-soar"
    model_path.mkdir(parents=True)
    for name in adapter.MODEL_FILES:
        (model_path / name).write_text('{}' if name == "config.json" else name)
    source = root / "apps/api/src/t1000"
    source.mkdir(parents=True)
    for name in adapter.PROFILE_SOURCES["low-vram"]:
        (source / name).write_text(name)
    monkeypatch.setitem(adapter.MODEL_PINS, "soar", {
        **adapter.MODEL_PINS["soar"], "sha256": sha256(model_path / "model.safetensors"),
    })
    sentence = T1000Backend(tmp_path, python_executable=Path(sys.executable), t1000_root=root)
    raw = T1000Backend(tmp_path, python_executable=Path(sys.executable), t1000_root=root, reference_boundary="raw")
    for model in (sentence, raw):
        model._runtime_info = {"verified_test_environment": True}
    assert sentence.signature() != raw.signature()


def test_decode_observer_distinguishes_eos_from_censored_cap():
    from types import SimpleNamespace

    for end_flag, expected in [(True, "eos"), (False, "schedule_exhausted")]:
        model = SimpleNamespace(_decode=lambda **kwargs: iter(["one", "two"]))
        observer = worker_module().DecodeObserver(model)
        assert list(observer(state=SimpleNamespace(end_flag=end_flag))) == ["one", "two"]
        assert observer.stop_reason == expected


def test_boundary_branch_preserves_native_speech_and_refuses_loud_excluded_edges(tmp_path, monkeypatch):
    from types import SimpleNamespace

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("boundary_test", scripts / "benchmark_prompt_boundaries.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    class Detector:
        def __init__(self, *args, **kwargs):
            self.queue = []

        def accept_waveform(self, values):
            pass

        def flush(self):
            self.queue.append(SimpleNamespace(start=4032, samples=np.zeros(23488)))

        def empty(self):
            return not self.queue

        @property
        def front(self):
            return self.queue[0]

        def pop(self):
            self.queue.pop(0)

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(
        VadModelConfig=lambda: SimpleNamespace(silero_vad=SimpleNamespace()), VoiceActivityDetector=Detector,
    ))
    monkeypatch.setattr(script, "verified_model_path", lambda root, name: tmp_path / "fake-vad.onnx")
    samples = np.zeros(129536, dtype=np.float32)
    samples[12096:82560] = np.sin(np.arange(82560 - 12096) * .02) * .2
    path = tmp_path / "original.wav"
    sf.write(path, samples, 48000, subtype="FLOAT")
    original = path.read_bytes()
    branch, rate, info = script.speech_edge_branch(path, tmp_path)
    assert rate == 48000 and info["selected_native_frames"] == [4896, 89760]
    np.testing.assert_array_equal(branch, samples[4896:89760])
    assert path.read_bytes() == original
    samples[90000:] = .5
    sf.write(path, samples, 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="not clearly quiet"):
        script.speech_edge_branch(path, tmp_path)


@pytest.mark.parametrize("conditions", [[], ["--conditions", "period-only", "--candidate-label", "MF2/reviewed-text"]])
def test_boundary_comparison_generates_fixed_candidates_without_extending_controls(tmp_path, monkeypatch, conditions):
    from t2000.backends.base import AudioChunk
    from t2000.profiles import ProfileStore
    from t2000.quality import audio_metrics, error_metrics
    from t2000.quality_models import sha256

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("boundary_loop_test", scripts / "benchmark_prompt_boundaries.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script.reporting, "BASELINE", script.reporting.BASELINE)
    monkeypatch.setenv("T2000_DATA_DIR", str(tmp_path / "data"))
    source = tmp_path / "reference.wav"
    pcm = (np.sin(np.arange(48000) * .03) * .2).astype(np.float32)
    sf.write(source, pcm, 48000, subtype="FLOAT")
    store = ProfileStore(tmp_path / "data/profiles")
    profile = store.create(source, "Own voice", "Hello, this is my voice")
    reference = store.reference_path(profile["id"])
    controls = [{"variant": script.BASELINE, "profile_id": profile["id"], "reference_path": str(reference),
                 "reference_sha256": sha256(reference), "case": case, "text": text, "expected_text": text,
                 "seed": 42, "complete": True, "score_status": "pending", "wav_path": str(source),
                 "audio": audio_metrics(source), "metrics": {"clipped_fraction": 0}}
                for case, text in [("one", "A natural sentence."), ("two", "A natural question?")]]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"records": controls}))
    generated = []

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            pass

        def prepare(self, path, cache, *, reference_text):
            assert reference_text == "Hello, this is my voice."

        def synthesize(self, text, cache, *, speed, seed):
            generated.append(text)
            return AudioChunk(pcm, 48000, .1)

        def info(self):
            return {"last_worker": {"stop_reason": "eos"}}

        def close(self):
            pass

    def score(report, settings, output):
        for row in report["records"]:
            row.update(score_status="ok", asr=error_metrics(row["expected_text"], row["expected_text"]),
                       speaker_cosine=.5)

    monkeypatch.setattr(script, "T1000Backend", FakeBackend)
    monkeypatch.setattr(script, "speech_edge_branch", lambda *args: (pcm, 48000, {"test": True}))
    monkeypatch.setattr(script.reporting, "score_outputs", score)
    output = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", ["benchmark_prompt_boundaries", "--t1000-root", str(tmp_path),
                                     "--profile", profile["id"], "--baseline-report", str(baseline),
                                     "--output-dir", str(output), *conditions])
    assert script.main() == 0
    report = json.loads((output / "report.json").read_text())
    assert len(report["records"]) == (4 if conditions else 6)  # Two controls plus two rows per condition.
    assert len(generated) == (3 if conditions else 6)  # Includes one excluded warmup per condition.
    assert all(summary["requests"] == 2 for summary in report["summaries"].values())
    assert store.get(profile["id"])["transcript"] == "Hello, this is my voice"

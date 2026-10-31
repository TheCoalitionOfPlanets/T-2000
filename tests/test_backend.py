"""Adapter boundary tests: no checkpoints, optional ML packages, or network needed."""

from contextlib import nullcontext
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import soundfile as sf

from t2000.backends.melo_openvoice import MeloOpenVoiceBackend
from t2000.errors import T2000Error


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def to(self, device):
        return self

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.values, axis))

    def transpose(self, first, second):
        return Tensor(self.values.swapaxes(first, second))

    def size(self, axis):
        return self.values.shape[axis]

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def mean(self, axis):
        return Tensor(self.values.mean(axis=axis))

    def numpy(self):
        return self.values


class TorchBoundary:
    def __init__(self):
        self.loads = []
        self.seeds = []
        self.random = SimpleNamespace(fork_rng=lambda **kwargs: nullcontext())

    from_numpy = staticmethod(Tensor)
    LongTensor = staticmethod(Tensor)
    inference_mode = staticmethod(nullcontext)
    is_tensor = staticmethod(lambda value: isinstance(value, Tensor))
    isfinite = staticmethod(lambda value: np.isfinite(value.values))
    stack = staticmethod(lambda values: Tensor(np.stack([value.values for value in values])))

    def manual_seed(self, seed):
        self.seeds.append(seed)

    def save(self, value, path):
        with Path(path).open("wb") as target:
            pickle.dump(value, target)

    def load(self, path, *, map_location, weights_only):
        self.loads.append((Path(path), map_location, weights_only))
        # This is a fake upstream boundary, never production checkpoint loading.
        with Path(path).open("rb") as source:
            return pickle.load(source)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    root = tmp_path / "models"
    required = ["melo/config.json", "melo/checkpoint.pth", "openvoice/converter/config.json",
                "openvoice/converter/checkpoint.pth", "openvoice/base_speakers/ses/en-us.pth",
                "watermark/model.pkl", "sources/MeloTTS/melo/api.py", "sources/OpenVoice/openvoice/api.py"]
    files = {}
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({"schema": 1, "complete": True, "files": files}))
    model = MeloOpenVoiceBackend(root)
    torch = TorchBoundary()
    calls = SimpleNamespace(load=0, reference=0, tts=[], conversion=[], watermark=[], spec_samples=[])
    config = SimpleNamespace(sampling_rate=22050, filter_length=1024, hop_length=256, win_length=1024)

    def ref_enc(spec):
        calls.reference += 1
        return Tensor(np.full((1, 256), calls.reference, dtype=np.float32))

    def conversion(spec, lengths, **kwargs):
        calls.conversion.append((spec, lengths, kwargs))
        return (Tensor(np.full((1, 1, 20000), 0.2, dtype=np.float32)),)

    def watermark(samples, message):
        calls.watermark.append(message)
        return samples

    converter = SimpleNamespace(hps=SimpleNamespace(data=config), add_watermark=watermark,
                                model=SimpleNamespace(ref_enc=ref_enc, voice_conversion=conversion))

    def tts_to_file(text, speaker_id, *, output_path, speed, quiet):
        calls.tts.append((text, speaker_id, output_path, speed, quiet))
        return np.full(44100, 0.1, dtype=np.float32)

    tts = SimpleNamespace(hps=SimpleNamespace(data=SimpleNamespace(sampling_rate=44100, spk2id={"EN-US": 7})),
                          tts_to_file=tts_to_file)

    def spec(tensor, n_fft, rate, hop, window, *, center):
        calls.spec_samples.append(tensor.values.copy())
        assert (n_fft, rate, hop, window, center) == (1024, 22050, 256, 1024, False)
        return Tensor(np.ones((1, 513, max(1, tensor.shape[-1] // 256)), dtype=np.float32))

    def initialize():
        calls.load += 1
        return torch, tts, converter, Tensor(np.zeros((1, 256, 1))), spec

    monkeypatch.setattr(model, "_initialize_runtime", initialize)
    reference = tmp_path / "reference.wav"
    # The native-rate, stereo input requires a separate mono/resampled branch.
    sf.write(reference, np.full((48000, 2), 0.25, dtype=np.float32), 48000, subtype="FLOAT")
    return model, calls, torch, reference, tmp_path / "profile"


def test_real_adapter_contract_caches_target_and_converts_in_memory(backend):
    model, calls, torch, reference, cache = backend
    model.prepare(reference, cache)
    model.prepare(reference, cache)
    assert calls.load == 1
    assert calls.reference == 1
    assert calls.spec_samples[0].shape == (1, 22050)
    assert not list(cache.glob("*.tmp"))
    result = model.synthesize("A novel test sentence.", cache, speed=1.25, seed=42)
    assert calls.tts == [("A novel test sentence.", 7, None, 1.25, True)]
    assert calls.spec_samples[-1].shape == (1, 22050)
    assert calls.conversion[0][2]["tau"] == 0.3
    assert calls.conversion[0][2]["sid_src"] is model._source
    assert calls.watermark == ["@MyShell"]
    assert torch.seeds == [42]
    assert all(location == "cpu" and weights_only for _, location, weights_only in torch.loads)
    assert result.samples.dtype == np.float32
    assert result.samples.ndim == 1 and result.sample_rate == 22050
    assert result.base_seconds >= 0 and result.conversion_seconds >= 0


def test_changed_reference_recomputes_conditioning(backend):
    model, calls, _, reference, cache = backend
    model.prepare(reference, cache)
    sf.write(reference, np.full(48000, 0.5, dtype=np.float32), 48000, subtype="FLOAT")
    model.prepare(reference, cache)
    assert calls.reference == 2


def test_conversion_tau_is_explicit_and_changes_conditioning_signature(backend):
    model, calls, _, reference, cache = backend
    default = model.signature()
    model.tau = 0.1
    assert model.signature() != default
    model.prepare(reference, cache)
    model.synthesize("A controlled conversion candidate.", cache, speed=1, seed=42)
    assert calls.conversion[0][2]["tau"] == 0.1


@pytest.mark.parametrize("kwargs", [{"reference_mode": "unknown"}, {"tau": float("nan")},
                                   {"tau": -0.1}, {"tau": 1.1}, {"tau": True}])
def test_invalid_reference_candidate_settings_are_rejected(tmp_path, kwargs):
    with pytest.raises(T2000Error, match="Reference mode|tau"):
        MeloOpenVoiceBackend(tmp_path, **kwargs)


def speech_mode(model, monkeypatch, tmp_path):
    model.reference_mode = "speech"
    vad_path = tmp_path / "pinned-vad.onnx"
    vad_path.write_bytes(b"the VAD verifier boundary")
    monkeypatch.setattr(model, "_vad_path", lambda: vad_path)
    return vad_path


def test_speech_reference_averages_chunk_embeddings_and_preserves_recording(backend, monkeypatch, tmp_path):
    model, calls, torch, reference, cache = backend
    vad_path = speech_mode(model, monkeypatch, tmp_path)
    sf.write(reference, np.full(26 * 22050, 0.2, dtype=np.float32), 22050, subtype="FLOAT")
    original = reference.read_bytes()
    monkeypatch.setattr(model, "_speech_regions", lambda samples: [(0, len(samples))])
    signature = model.signature()
    model.prepare(reference, cache)
    assert calls.reference == 3
    assert max(values.shape[-1] for values in calls.spec_samples) <= 10 * 22050
    saved = torch.load(cache / "target.pt", map_location="cpu", weights_only=True)
    np.testing.assert_allclose(saved["embedding"].values, 2)
    assert saved["conditioning"]["segments"] == 3
    assert saved["conditioning"]["conditioning_seconds"] == 26
    assert reference.read_bytes() == original
    model.prepare(reference, cache)
    assert calls.reference == 3  # Complete speech-only conditioning is reused.
    vad_path.write_bytes(b"changed detector")
    assert model.signature() != signature


def test_short_speech_reference_is_one_segment_and_no_speech_is_actionable(backend, monkeypatch, tmp_path):
    model, calls, _, reference, cache = backend
    speech_mode(model, monkeypatch, tmp_path)
    model._sample_rate = 22050
    monkeypatch.setattr(model, "_speech_regions", lambda samples: [(0, len(samples))])
    samples = np.full(round(3.904 * 22050), 0.2, dtype=np.float32)
    segments, info = model._reference_segments(samples)
    assert len(segments) == 1 and info["segments"] == 1
    np.testing.assert_array_equal(segments[0], samples)
    monkeypatch.setattr(model, "_speech_regions", lambda samples: [])
    with pytest.raises(T2000Error, match="No speech was detected"):
        model.prepare(reference, cache)
    assert calls.reference == 0 and not (cache / "target.pt").exists()


def test_vad_flush_maps_padded_regions_to_original_rate_and_merges_overlap(backend, monkeypatch, tmp_path):
    import sys

    model, _, _, _, _ = backend
    speech_mode(model, monkeypatch, tmp_path)
    model._sample_rate = 22050
    detectors = []

    class Detector:
        def __init__(self, config, *, buffer_size_in_seconds):
            self.config, self.chunks, self.queue = config, [], []
            detectors.append(self)

        def accept_waveform(self, samples):
            self.chunks.append(np.asarray(samples))

        def empty(self):
            return not self.queue

        @property
        def front(self):
            return self.queue[0]

        def pop(self):
            self.queue.pop(0)

        def flush(self):
            # Both intervals appear only on flush, including the recording tail.
            self.queue.extend([SimpleNamespace(start=3200, samples=np.zeros(3200)),
                               SimpleNamespace(start=7200, samples=np.zeros(8800))])

    module = SimpleNamespace(
        VadModelConfig=lambda: SimpleNamespace(silero_vad=SimpleNamespace()),
        VoiceActivityDetector=Detector,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    samples = np.linspace(-0.2, 0.2, 22050, dtype=np.float32)
    original = samples.copy()
    regions = model._speech_regions(samples)
    assert len(regions) == 1
    assert regions[0] == (2205, len(samples))  # 0.2s speech start minus 0.1s padding.
    assert sum(len(chunk) for chunk in detectors[0].chunks) == 16000
    assert max(len(chunk) for chunk in detectors[0].chunks) == 512
    assert detectors[0].config.sample_rate == 16000
    np.testing.assert_array_equal(samples, original)


def test_model_mutation_is_rejected_and_settings_change_signature(backend):
    model, _, _, _, _ = backend
    signature = model.signature()
    assert MeloOpenVoiceBackend(model.model_dir, num_threads=3).signature() != signature
    (model.model_dir / "melo/checkpoint.pth").write_bytes(b"unexpected checkpoint")
    with pytest.raises(T2000Error, match="integrity"):
        model.load()


def test_corrupt_conditioning_is_actionable_and_never_emits_fallback(backend):
    model, calls, _, reference, cache = backend
    model.prepare(reference, cache)
    (cache / "target.pt").write_bytes(b"invalid conditioning")
    model._profiles.clear()
    with pytest.raises(T2000Error, match="Enroll the reference again"):
        model.synthesize("Hello.", cache, speed=1.0, seed=1)
    assert not calls.tts


def test_profile_from_another_signature_is_rejected(backend):
    model, calls, torch, reference, cache = backend
    model.prepare(reference, cache)
    saved = torch.load(cache / "target.pt", map_location="cpu", weights_only=True)
    saved["signature"] = "another model"
    torch.save(saved, cache / "target.pt")
    model._profiles.clear()
    with pytest.raises(T2000Error, match="different voice models"):
        model.synthesize("Hello.", cache, speed=1.0, seed=1)
    assert not calls.tts


def test_missing_models_and_invalid_controls_fail_without_runtime_import(tmp_path):
    model = MeloOpenVoiceBackend(tmp_path)
    assert not model.info()["loaded"]
    with pytest.raises(T2000Error, match="setup_models"):
        model.load()
    with pytest.raises(T2000Error, match="speed"):
        model.synthesize("Hello.", tmp_path, speed=float("nan"), seed=0)


@pytest.mark.parametrize("manifest", [[], {"schema": 1, "complete": True, "files": ["a"]},
                                     {"schema": 1, "complete": True, "files": {"a": None}}])
def test_malformed_manifest_is_actionable(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(T2000Error, match="setup_models"):
        MeloOpenVoiceBackend(tmp_path).signature()


@pytest.fixture
def installer():
    path = Path(__file__).resolve().parents[1] / "scripts/setup_models.py"
    spec = importlib.util.spec_from_file_location("model_setup_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nlp_fetch_falls_back_from_blob_api_and_reuses_verified_archive(installer, tmp_path, monkeypatch):
    payload = b"a pinned NLP resource"
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    metadata = {"sha": blob, "size": len(payload), "encoding": "base64",
                "content": base64.b64encode(payload).decode()}
    requests = []

    def urlopen(request, **kwargs):
        requests.append(request.full_url)
        if "/git/blobs/" in request.full_url:
            raise TimeoutError("simulated CDN/API timeout")
        return io.BytesIO(json.dumps(metadata).encode())

    monkeypatch.setattr(installer.urllib.request, "urlopen", urlopen)
    target = tmp_path / "resource.zip"
    for _ in range(2):
        installer.github_blob("nltk/nltk_data", "pinned", "packages/resource.zip", target,
                              expected_git_blob=blob, expected_size=len(payload))
    assert target.read_bytes() == payload
    assert len(requests) == 2  # The second invocation is fully offline.
    assert "ref=pinned" in requests[-1]
    assert not list(tmp_path.glob("*.part"))


def test_nlp_fetch_rejects_tampered_bytes_on_every_route(installer, tmp_path, monkeypatch):
    expected = b"the expected resource"
    bad = b"!" + expected[1:]
    blob = hashlib.sha1(f"blob {len(expected)}\0".encode() + expected).hexdigest()
    metadata = {"sha": blob, "size": len(expected), "encoding": "base64",
                "content": base64.b64encode(bad).decode()}

    def urlopen(request, **kwargs):
        if "api.github.com" in request.full_url:
            return io.BytesIO(json.dumps(metadata).encode())
        return io.BytesIO(bad)

    monkeypatch.setattr(installer.urllib.request, "urlopen", urlopen)
    with pytest.raises(ValueError, match="checksum mismatch"):
        installer.github_blob("nltk/nltk_data", "pinned", "packages/resource.zip", tmp_path / "resource.zip",
                              expected_git_blob=blob, expected_size=len(expected))
    assert not list(tmp_path.iterdir())


def test_setup_rejects_archive_traversal_before_writing(installer, tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("../outside.txt", "unexpected")
    with pytest.raises(ValueError, match="unsafe archive path"):
        installer.extract(archive, tmp_path / "models")
    assert not (tmp_path / "outside.txt").exists()


def test_nlp_install_retains_legacy_zip_and_extracted_resource(installer, tmp_path):
    archive = tmp_path / "verified-download.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("cmudict/cmudict", "a small test dictionary")
    destination = tmp_path / "nltk_data/corpora"
    installer.install_nlp_archive(archive, destination, "cmudict")
    assert (destination / "cmudict.zip").read_bytes() == archive.read_bytes()
    assert (destination / "cmudict/cmudict").read_text() == "a small test dictionary"
    assert not list(destination.glob("*.part"))


def test_runtime_preflights_g2p_en_literal_zip_paths_before_import(tmp_path, monkeypatch):
    calls = []
    nltk = SimpleNamespace(data=SimpleNamespace(path=[], find=calls.append))
    torch = SimpleNamespace(set_num_threads=lambda value: None)

    class PreflightFinished(Exception):
        pass

    def import_module(name):
        if name == "torch":
            return torch
        if name == "nltk":
            return nltk
        if name == "melo.api":
            raise PreflightFinished()
        raise AssertionError(f"Unexpected import {name}")

    # Keep the adapter's process environment/path configuration inside this test.
    import os
    import sys
    import t2000.backends.melo_openvoice as adapter
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(adapter.importlib, "import_module", import_module)
    model = MeloOpenVoiceBackend(tmp_path)
    with pytest.raises(PreflightFinished):
        model._initialize_runtime()
    assert "corpora/cmudict.zip" in calls
    assert "taggers/averaged_perceptron_tagger.zip" in calls
    assert "taggers/averaged_perceptron_tagger_eng" in calls


def test_runtime_constructs_converter_without_upstream_watermark_keyword_bug(tmp_path, monkeypatch):
    import os
    import sys
    import t2000.backends.melo_openvoice as adapter

    base_initializations, checkpoint_loads = [], []

    class Network:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state, *, strict):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    class Module:
        def __init__(self):
            pass

    class TTS(Module):
        pass

    class BaseConverter:
        # Exact strict signature that rejects enable_watermark in pinned upstream.
        def __init__(self, config_path, device="cpu"):
            base_initializations.append((config_path, device))
            self.hps = SimpleNamespace(_version_="v2")
            self.model, self.device = Network(), device

    class Converter(BaseConverter):
        def __init__(self, *args, **kwargs):
            raise AssertionError("The broken upstream convenience initializer must be bypassed")

    hps = SimpleNamespace(symbols=["_", "a"], num_tones=16, num_languages=10, model={},
                          data=SimpleNamespace(filter_length=2048, hop_length=512, n_speakers=256),
                          train=SimpleNamespace(segment_size=16384))
    source_embedding = Tensor(np.zeros((1, 256, 1), dtype=np.float32))

    def checkpoint(path, *, map_location, weights_only):
        checkpoint_loads.append((str(path), map_location, weights_only))
        return source_embedding if "/ses/" in str(path) else {"model": {}}

    torch = SimpleNamespace(set_num_threads=lambda value: None, nn=SimpleNamespace(Module=Module),
                            load=checkpoint)
    nltk = SimpleNamespace(data=SimpleNamespace(path=[], find=lambda resource: None))
    melo = SimpleNamespace(TTS=TTS, SynthesizerTrn=Network,
                           utils=SimpleNamespace(get_hparams_from_file=lambda path: hps))
    ov = SimpleNamespace(ToneColorConverter=Converter, OpenVoiceBaseClass=BaseConverter,
                         spectrogram_torch=lambda *args, **kwargs: None)
    modules = {"torch": torch, "nltk": nltk, "melo.api": melo, "openvoice.api": ov,
               "wavmark.models.my_model": SimpleNamespace(Model=Network)}
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for name, folder in [("melo", "MeloTTS"), ("openvoice", "OpenVoice")]:
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(
            __file__=str(tmp_path / "sources" / folder / name / "__init__.py")))
    monkeypatch.setattr(adapter.importlib, "import_module", modules.__getitem__)
    model = MeloOpenVoiceBackend(tmp_path)
    _, _, converter, source, _ = model._initialize_runtime()
    assert base_initializations == [(str(tmp_path / "openvoice/converter/config.json"), "cpu")]
    assert converter.version == "v2"
    assert isinstance(converter.watermark_model, Network)
    assert source is source_embedding
    assert len(checkpoint_loads) == 4
    assert all(location == "cpu" and weights_only for _, location, weights_only in checkpoint_loads)

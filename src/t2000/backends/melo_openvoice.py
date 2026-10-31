"""Local MeloTTS/OpenVoice V2 adapter. Imports the optional stack only on load.

The upstream converter accepts whole utterances. The engine may schedule short
phrases, but this adapter does not claim frame-level streaming conversion.
"""

from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from time import perf_counter

import numpy as np

from .base import AudioChunk
from ..errors import T2000Error


ADAPTER_VERSION = "melo-openvoice-v2-speech-profile-2"
WATERMARK_MESSAGE = "@MyShell"
REFERENCE_POLICY = {
    "version": "silero-speech-segments-v1", "sample_rate": 16000, "window_size": 512,
    "threshold": 0.5, "min_silence_duration": 1.0, "min_speech_duration": 0.1,
    "max_speech_duration": 30.0, "padding_seconds": 0.1, "segment_seconds": 10.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MeloOpenVoiceBackend:
    def __init__(self, model_dir: Path, tts_device="cpu", converter_device="cpu",
                 speaker="EN-US", language="EN", num_threads=4, reference_mode="full", tau=0.3):
        if reference_mode not in {"full", "speech"}:
            raise T2000Error("Reference mode must be full or speech.")
        if (isinstance(tau, bool) or not isinstance(tau, (int, float))
                or not np.isfinite(tau) or not 0 <= tau <= 1):
            raise T2000Error("Converter tau must be a finite number between 0 and 1.")
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.tts_device = tts_device
        self.converter_device = converter_device
        self.speaker = speaker.upper().replace("_", "-")
        self.language = language.upper()
        self.num_threads = num_threads
        self.reference_mode = reference_mode
        self.tau = float(tau)
        self._loaded = False
        self._hash_cache: dict[str, tuple[tuple[int, int, int], str]] = {}
        self._profiles: dict[str, tuple[tuple[int, int], object]] = {}
        self._loaded_signature: str | None = None

    def _manifest(self):
        try:
            manifest = json.loads((self.model_dir / "manifest.json").read_text())
            if (not isinstance(manifest, dict) or manifest.get("schema") != 1
                    or not manifest.get("complete") or not isinstance(manifest.get("files"), dict)
                    or not manifest["files"]):
                raise ValueError("incomplete or unsupported manifest")
            for relative, digest in manifest["files"].items():
                if not isinstance(relative, str) or not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError("invalid file checksum entry")
                int(digest, 16)
            return manifest
        except (OSError, ValueError, TypeError) as exc:
            raise T2000Error(
                "Voice models are not prepared. Run python scripts/setup_models.py "
                f"--model-dir {self.model_dir}; see docs/models.md."
            ) from exc

    def signature(self) -> str:
        manifest = self._manifest()
        actual = {}
        try:
            for relative, expected in sorted(manifest["files"].items()):
                path = (self.model_dir / relative).resolve()
                if not path.is_relative_to(self.model_dir):
                    raise ValueError("manifest contains a path outside the model directory")
                stat = path.stat()
                stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                cached = self._hash_cache.get(relative)
                digest = cached[1] if cached and cached[0] == stamp else _sha256(path)
                self._hash_cache[relative] = (stamp, digest)
                if digest != expected:
                    raise ValueError(f"checksum mismatch for {relative}")
                actual[relative] = digest
        except (OSError, ValueError, TypeError) as exc:
            raise T2000Error(f"Voice model integrity check failed: {exc}. Run setup_models.py --check.") from exc
        required = ["melo/config.json", "melo/checkpoint.pth", "openvoice/converter/config.json",
                    "openvoice/converter/checkpoint.pth",
                    f"openvoice/base_speakers/ses/{self.speaker.lower()}.pth", "watermark/model.pkl",
                    "sources/MeloTTS/melo/api.py", "sources/OpenVoice/openvoice/api.py"]
        if any(relative not in actual for relative in required):
            raise T2000Error("The model manifest is missing this base speaker or required model files. "
                             "Run scripts/setup_models.py again.")
        identity = {"adapter": ADAPTER_VERSION, "files": actual, "speaker": self.speaker,
                    "language": self.language, "tts_device": self.tts_device,
                    "converter_device": self.converter_device, "num_threads": self.num_threads,
                    "tau": self.tau, "watermark": WATERMARK_MESSAGE,
                    "reference_mode": self.reference_mode}
        if self.reference_mode == "speech":
            identity["reference_policy"] = REFERENCE_POLICY
            identity["vad_sha256"] = _sha256(self._vad_path())
        identity["runtime_versions"] = {}
        for package in ("torch", "numpy", "librosa", "soxr", "transformers", "wavmark", "numba", "nltk",
                        "setuptools", "g2p-en", "tokenizers", "sherpa-onnx"):
            try:
                identity["runtime_versions"][package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                identity["runtime_versions"][package] = None
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def _initialize_runtime(self):
        """Construct upstream networks while keeping checkpoint loading explicit.

        Upstream convenience constructors call torch.load without weights_only;
        constructing Melo's network here avoids changing torch.load globally.
        """
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_CACHE"] = str(self.model_dir / "hf")
        os.environ["TRANSFORMERS_CACHE"] = str(self.model_dir / "hf")
        os.environ["NLTK_DATA"] = str(self.model_dir / "nltk_data")
        for name, package in [("MeloTTS", "melo"), ("OpenVoice", "openvoice")]:
            root = self.model_dir / "sources" / name
            existing = sys.modules.get(package)
            if existing and not Path(existing.__file__).resolve().is_relative_to(root):
                raise T2000Error(f"{package} was imported from another installation. Restart this process.")
            sys.path.insert(0, str(root))
        torch = importlib.import_module("torch")
        for device in (self.tts_device, self.converter_device):
            if device != "cpu" and not (device == "cuda" or device.startswith("cuda:")):
                raise T2000Error("Use cpu, cuda, or cuda:N for voice model devices.")
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise T2000Error("CUDA is unavailable. Set both voice devices to cpu or install GPU requirements.")
        torch.set_num_threads(self.num_threads)
        nltk = importlib.import_module("nltk")
        nltk.data.path[:] = [str(self.model_dir / "nltk_data")]
        for resource in ("corpora/cmudict", "corpora/cmudict.zip",
                         "taggers/averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger.zip",
                         "taggers/averaged_perceptron_tagger_eng", "tokenizers/punkt", "tokenizers/punkt_tab"):
            nltk.data.find(resource)  # Fail before g2p_en could try nltk.download.
        api = importlib.import_module("melo.api")
        hps = api.utils.get_hparams_from_file(str(self.model_dir / "melo/config.json"))
        tts = api.TTS.__new__(api.TTS)
        torch.nn.Module.__init__(tts)
        tts.model = api.SynthesizerTrn(
            len(hps.symbols), hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers, num_tones=hps.num_tones,
            num_languages=hps.num_languages, **hps.model,
        ).to(self.tts_device).eval()
        checkpoint = torch.load(self.model_dir / "melo/checkpoint.pth", map_location="cpu", weights_only=True)
        tts.model.load_state_dict(checkpoint["model"], strict=True)
        tts.hps, tts.device, tts.language = hps, self.tts_device, self.language
        tts.symbol_to_id = {symbol: index for index, symbol in enumerate(hps.symbols)}
        ov = importlib.import_module("openvoice.api")
        # The pinned convenience constructor forwards enable_watermark to a base
        # constructor that rejects it. Initialize that same base network directly,
        # then attach the verified local watermark below, without a hidden fetch.
        converter = ov.ToneColorConverter.__new__(ov.ToneColorConverter)
        ov.OpenVoiceBaseClass.__init__(
            converter, str(self.model_dir / "openvoice/converter/config.json"), device=self.converter_device,
        )
        converter.version = getattr(converter.hps, "_version_", "v1")
        checkpoint = torch.load(self.model_dir / "openvoice/converter/checkpoint.pth",
                                map_location="cpu", weights_only=True)
        result = converter.model.load_state_dict(checkpoint["model"], strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise T2000Error("The OpenVoice checkpoint does not match the pinned converter source/config.")
        watermark = importlib.import_module("wavmark.models.my_model").Model(
            16000, num_bit=32, n_fft=1000, hop_length=400, num_layers=8,
        )
        watermark.load_state_dict(torch.load(self.model_dir / "watermark/model.pkl",
                                             map_location="cpu", weights_only=True), strict=True)
        converter.watermark_model = watermark.to(self.converter_device).eval()
        source = torch.load(self.model_dir / f"openvoice/base_speakers/ses/{self.speaker.lower()}.pth",
                            map_location="cpu", weights_only=True).to(self.converter_device)
        return torch, tts, converter, source, ov.spectrogram_torch

    def load(self) -> None:
        if self.language != "EN":
            raise T2000Error("The prepared backend currently supports English (EN) only.")
        identity = self.signature()
        if self._loaded:
            if identity != self._loaded_signature:
                raise T2000Error("Voice model files/settings changed after load. Restart before generating.")
            return
        try:
            self._torch, self._tts, self._converter, self._source, self._spectrogram = self._initialize_runtime()
            speakers = self._tts.hps.data.spk2id
            matched = {key.upper().replace("_", "-"): value for key, value in speakers.items()}
            self._speaker_id = matched[self.speaker]
            self._sample_rate = int(self._converter.hps.data.sampling_rate)
            self._loaded_signature, self._loaded = identity, True
        except T2000Error:
            raise
        except Exception as exc:
            raise T2000Error(f"Could not load the local voice models: {exc}. "
                             "Install the inference requirements and complete scripts/setup_models.py.") from exc

    def _spec(self, samples: np.ndarray):
        hps = self._converter.hps.data
        tensor = self._torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
        tensor = tensor.to(self.converter_device).unsqueeze(0)
        return self._spectrogram(tensor, hps.filter_length, hps.sampling_rate,
                                 hps.hop_length, hps.win_length, center=False)

    def _resample(self, samples, source_rate):
        samples = np.asarray(samples, dtype=np.float32)
        if source_rate != self._sample_rate:
            import soxr
            samples = soxr.resample(samples, source_rate, self._sample_rate, quality="HQ")
        return np.ascontiguousarray(samples, dtype=np.float32)

    def _vad_path(self):
        from ..quality_models import verified_model_path

        return verified_model_path(self.model_dir, "silero_vad.onnx")

    def _speech_regions(self, samples: np.ndarray) -> list[tuple[int, int]]:
        """Detect speech at 16 kHz, returning padded converter-rate intervals.

        Detection never changes the enrollment WAV or uses the detector's own
        resampled samples as speaker-encoder input. The exact VAD asset is
        verified locally; importing this adapter cannot download it.
        """
        import soxr

        vad_path = self._vad_path()
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise T2000Error("Speech-only references require the STT dependencies. "
                             "Install the project's stt extra before preparing this voice.") from exc
        policy = REFERENCE_POLICY
        rate = policy["sample_rate"]
        if samples.size > 30 * self._sample_rate:
            raise T2000Error("Use a reference recording of 30 seconds or less.")
        detection = np.ascontiguousarray(
            soxr.resample(samples, self._sample_rate, rate, quality="HQ"), dtype=np.float32,
        )
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(vad_path)
        for name in ("threshold", "min_silence_duration", "min_speech_duration",
                     "max_speech_duration", "window_size"):
            setattr(config.silero_vad, name, policy[name])
        config.sample_rate = rate
        config.num_threads = min(self.num_threads, 2)
        config.provider = "cpu"
        detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=35)
        intervals = []

        def drain():
            while not detector.empty():
                segment = detector.front
                start = max(0, int((segment.start / rate - policy["padding_seconds"]) * self._sample_rate))
                end = min(samples.size, int(np.ceil(
                    ((segment.start + len(segment.samples)) / rate + policy["padding_seconds"])
                    * self._sample_rate,
                )))
                if end > start:
                    intervals.append((start, end))
                detector.pop()

        window = policy["window_size"]
        for offset in range(0, detection.size, window):
            detector.accept_waveform(detection[offset:offset + window])
            drain()
        detector.flush()
        drain()
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        return merged

    def _reference_segments(self, samples: np.ndarray) -> tuple[list[np.ndarray], dict]:
        if self.reference_mode == "full":
            return [samples], {"mode": "full", "segments": 1,
                               "conditioning_seconds": samples.size / self._sample_rate}
        intervals = self._speech_regions(samples)
        if not intervals:
            raise T2000Error("No speech was detected in this reference. Record a clear single-speaker sentence.")
        active = np.concatenate([samples[start:end] for start, end in intervals])
        if active.size < self._sample_rate // 2:
            raise T2000Error("The reference needs at least half a second of detected speech.")
        # Match upstream's roughly ten-second, equal-duration chunks, while
        # accepting useful references shorter than the upstream round-to-zero.
        count = max(1, round(active.size / self._sample_rate / REFERENCE_POLICY["segment_seconds"]))
        segments = [np.ascontiguousarray(part) for part in np.array_split(active, count)]
        return segments, {"mode": "speech", "policy": REFERENCE_POLICY, "segments": len(segments),
                          "detected_regions": len(intervals),
                          "conditioning_seconds": active.size / self._sample_rate}

    def prepare(self, reference_path: Path, cache_dir: Path, *, reference_text: str = "") -> None:
        self.load()
        try:
            import soundfile as sf
            reference_path, cache_dir = Path(reference_path), Path(cache_dir)
            reference_hash = _sha256(reference_path)
            cache = cache_dir / "target.pt"
            if cache.is_file():
                saved = self._torch.load(cache, map_location="cpu", weights_only=True)
                if saved.get("signature") == self._loaded_signature and saved.get("reference_sha256") == reference_hash:
                    self._remember_profile(cache, saved["embedding"])
                    return
            samples, rate = sf.read(reference_path, dtype="float32", always_2d=True)
            samples = self._resample(samples.mean(axis=1), rate)
            if samples.size < self._sample_rate // 2 or not np.isfinite(samples).all():
                raise T2000Error("Use at least half a second of finite reference speech.")
            segments, conditioning = self._reference_segments(samples)
            with self._torch.inference_mode():
                embeddings = [self._converter.model.ref_enc(self._spec(segment).transpose(1, 2)).unsqueeze(-1)
                              for segment in segments]
                embedding = (embeddings[0] if len(embeddings) == 1
                             else self._torch.stack(embeddings).mean(0))
            embedding = embedding.detach().cpu()
            cache_dir.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix="target-", suffix=".tmp", dir=cache_dir)
            os.close(fd)
            try:
                self._torch.save({"embedding": embedding, "signature": self._loaded_signature,
                                  "reference_sha256": reference_hash, "conditioning": conditioning}, temp_name)
                os.replace(temp_name, cache)
            finally:
                Path(temp_name).unlink(missing_ok=True)
            self._remember_profile(cache, embedding)
        except T2000Error:
            raise
        except Exception as exc:
            raise T2000Error(f"Could not prepare speaker conditioning: {exc}. "
                             "Remove this profile's target.pt and enroll the reference again.") from exc

    def _remember_profile(self, cache, embedding):
        if (not self._torch.is_tensor(embedding) or tuple(embedding.shape) != tuple(self._source.shape)
                or not self._torch.isfinite(embedding).all()):
            raise T2000Error("Speaker conditioning is invalid. Enroll the reference again.")
        stat = cache.stat()
        self._profiles[str(cache.resolve())] = ((stat.st_mtime_ns, stat.st_size),
                                               embedding.to(self.converter_device))

    def _target(self, cache_dir):
        cache = Path(cache_dir) / "target.pt"
        try:
            stat = cache.stat()
            cached = self._profiles.get(str(cache.resolve()))
            if cached and cached[0] == (stat.st_mtime_ns, stat.st_size):
                return cached[1]
            saved = self._torch.load(cache, map_location="cpu", weights_only=True)
            if saved.get("signature") != self._loaded_signature:
                raise ValueError("conditioning was made with different voice models or settings")
            self._remember_profile(cache, saved["embedding"])
            return self._profiles[str(cache.resolve())][1]
        except T2000Error:
            raise
        except Exception as exc:
            raise T2000Error(f"Speaker conditioning is unavailable: {exc}. Enroll the reference again.") from exc

    def _synchronize(self, device):
        if device.startswith("cuda"):
            self._torch.cuda.synchronize(device)

    def synthesize(self, text: str, cache_dir: Path, *, speed: float, seed: int) -> AudioChunk:
        if not text.strip() or not np.isfinite(speed) or not 0.5 <= speed <= 2:
            raise T2000Error("Provide text and a speaking speed between 0.5 and 2.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise T2000Error("The seed must be an integer between 0 and 4294967295.")
        self.load()
        target = self._target(cache_dir)
        np_state, random_state = np.random.get_state(), random.getstate()
        cuda_devices = sorted({self._torch.device(d).index or 0 for d in (self.tts_device, self.converter_device)
                               if d.startswith("cuda")})
        try:
            with self._torch.random.fork_rng(devices=cuda_devices), self._torch.inference_mode():
                self._torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                self._synchronize(self.tts_device)
                start = perf_counter()
                base = self._tts.tts_to_file(text, self._speaker_id, output_path=None, speed=speed, quiet=True)
                self._synchronize(self.tts_device)
                samples = self._resample(base, int(self._tts.hps.data.sampling_rate))
                base_seconds = perf_counter() - start
                start = perf_counter()
                spec = self._spec(samples)
                lengths = self._torch.LongTensor([spec.size(-1)]).to(self.converter_device)
                converted = self._converter.model.voice_conversion(
                    spec, lengths, sid_src=self._source, sid_tgt=target, tau=self.tau,
                )[0][0, 0].detach().cpu().float().numpy()
                # Keep upstream's @MyShell watermark and short-audio behavior.
                converted = self._converter.add_watermark(converted, WATERMARK_MESSAGE)
                self._synchronize(self.converter_device)
                conversion_seconds = perf_counter() - start
            converted = np.ascontiguousarray(converted, dtype=np.float32)
            if converted.ndim != 1 or not converted.size or not np.isfinite(converted).all():
                raise T2000Error("The voice converter returned invalid audio.")
            return AudioChunk(converted, self._sample_rate, base_seconds, conversion_seconds)
        except T2000Error:
            raise
        except Exception as exc:
            raise T2000Error(f"Voice generation failed: {exc}. No substitute audio was generated.") from exc
        finally:
            np.random.set_state(np_state)
            random.setstate(random_state)

    def info(self) -> dict:
        return {"backend": "melo-openvoice-v2", "loaded": self._loaded, "model_dir": str(self.model_dir),
                "language": self.language, "speaker": self.speaker, "tts_device": self.tts_device,
                "converter_device": self.converter_device, "sample_rate": getattr(self, "_sample_rate", None),
                "reference_mode": self.reference_mode, "tau": self.tau,
                "signature": self._loaded_signature, "streaming": "complete phrases",
                "watermark": "upstream @MyShell; short phrases may be too short to contain all watermark blocks"}

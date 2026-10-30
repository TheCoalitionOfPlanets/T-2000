"""Optional CPU streaming Nemotron ASR, adapted from T-1000's STTSession.

Source: https://github.com/TheCoalitionOfPlanets/T-1000
Each recording owns resampling and decoder state; chunks are consumed once.
The optional sherpa-onnx import and weight loading happen only on transcription.
"""

from collections import OrderedDict
from pathlib import Path
import threading

import numpy as np
import soundfile as sf
import soxr

from .errors import T2000Error


MODEL_NAME = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
MODEL_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
SAMPLE_RATE = 16000
_LOAD_LOCK = threading.Lock()
_DECODE_LOCK = threading.Lock()
_recognizers = OrderedDict()


class STTError(T2000Error):
    """An actionable recognition setup or recording failure."""


def _load_recognizer(model_dir: Path, num_threads: int):
    model_dir = Path(model_dir).expanduser().resolve()
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
        raise STTError("STT thread count must be a positive integer.")
    key = (str(model_dir), num_threads)
    with _LOAD_LOCK:
        if key in _recognizers:
            _recognizers.move_to_end(key)
            return _recognizers[key]
        if any(not (model_dir / filename).is_file() for filename in MODEL_FILES):
            raise STTError("Speech recognition weights are missing. Run python scripts/fetch_stt.py "
                           f"--model-dir {model_dir} before transcribing.")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise STTError("Install the optional STT dependencies with pip install -e '.[stt]' before transcribing.") from exc
        try:
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(model_dir / "tokens.txt"), encoder=str(model_dir / "encoder.int8.onnx"),
                decoder=str(model_dir / "decoder.int8.onnx"), joiner=str(model_dir / "joiner.int8.onnx"),
                model_type="nemotron", sample_rate=SAMPLE_RATE, feature_dim=128,
                num_threads=num_threads, provider="cpu", decoding_method="greedy_search",
                enable_endpoint_detection=False,
            )
        except Exception as exc:
            raise STTError(f"Could not load speech recognition weights: {exc}") from exc
        _recognizers[key] = recognizer
        # Sessions retain their own recognizer references after eviction.
        while len(_recognizers) > 2:
            _recognizers.popitem(last=False)
        return recognizer


def _mono_float32(audio):
    values = np.asarray(audio)
    if values.ndim not in (1, 2) or (values.ndim == 2 and not 1 <= values.shape[1] <= 8):
        raise STTError("Audio must contain mono or stereo sample frames.")
    if np.issubdtype(values.dtype, np.signedinteger):
        values = values.astype(np.float32) / float(-np.iinfo(values.dtype).min)
    elif values.dtype == np.uint8:
        values = (values.astype(np.float32) - 128) / 128
    elif np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    else:
        raise STTError("Audio must contain numeric PCM or floating-point samples.")
    if not np.isfinite(values).all():
        raise STTError("The recording contains invalid audio samples.")
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float64).astype(np.float32)
    return np.ascontiguousarray(values)


class STTSession:
    """One incremental English transcription; finish is safe to repeat."""

    def __init__(self, model_dir: Path, num_threads: int = 4, max_seconds: float = 30):
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite")
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise STTError("STT thread count must be a positive integer.")
        self.model_dir, self.num_threads = Path(model_dir), num_threads
        self.max_seconds, self.text, self.finished = float(max_seconds), "", False
        self.audio_seconds, self._input_samples = 0.0, 0
        self._input_rate = self._resampler = self._recognizer = self._stream = None
        self._lock = threading.Lock()

    def _ensure_stream(self):
        if self._stream is None:
            self._recognizer = _load_recognizer(self.model_dir, self.num_threads)
            with _DECODE_LOCK:
                self._stream = self._recognizer.create_stream()
                self._stream.accept_waveform(SAMPLE_RATE, np.zeros(8000, dtype=np.float32))

    def _accept(self, samples):
        for offset in range(0, samples.size, SAMPLE_RATE):
            with _DECODE_LOCK:
                self._stream.accept_waveform(SAMPLE_RATE, samples[offset:offset + SAMPLE_RATE])
                while self._recognizer.is_ready(self._stream):
                    self._recognizer.decode_stream(self._stream)
                self.text = self._recognizer.get_result(self._stream)

    def feed(self, sample_rate: int, audio) -> str:
        with self._lock:
            if self.finished:
                raise STTError("This transcription has finished. Start a new session.")
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
                raise STTError("The recording's sample rate must be a positive integer.")
            if self._input_rate is not None and sample_rate != self._input_rate:
                raise STTError("The recording's sample rate changed. Start a new session.")
            samples = _mono_float32(audio)
            if self._input_samples + len(samples) > int(self.max_seconds * sample_rate):
                raise STTError(f"The recording exceeds the {self.max_seconds:g}-second transcription limit.")
            if not samples.size:
                return self.text
            self._ensure_stream()
            if self._input_rate is None:
                self._input_rate = int(sample_rate)
                if sample_rate != SAMPLE_RATE:
                    self._resampler = soxr.ResampleStream(sample_rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ")
            self._input_samples += len(samples)
            self.audio_seconds = self._input_samples / sample_rate
            if self._resampler is not None:
                samples = self._resampler.resample_chunk(samples)
            try:
                self._accept(samples)
            except Exception as exc:
                raise STTError(f"Speech recognition failed: {exc}") from exc
            return self.text

    def finish(self) -> str:
        with self._lock:
            if self.finished:
                return self.text
            try:
                if self._stream is not None:
                    if self._resampler is not None:
                        self._accept(self._resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True))
                    self._accept(np.zeros(12800, dtype=np.float32))
                    with _DECODE_LOCK:
                        self._stream.input_finished()
                        while self._recognizer.is_ready(self._stream):
                            self._recognizer.decode_stream(self._stream)
                        self.text = self._recognizer.get_result(self._stream)
                self.finished = True
                return self.text
            except Exception as exc:
                raise STTError(f"Could not finish speech recognition: {exc}") from exc

    def close(self):
        with self._lock:
            self.finished = True
            self._stream = self._recognizer = self._resampler = None


def transcribe_file(path: Path, model_dir: Path, num_threads: int = 4, max_seconds: float = 30) -> str:
    """Transcribe a whole bounded reference; no downloads or truncated transcript."""
    session = STTSession(model_dir, num_threads=num_threads, max_seconds=max_seconds)
    try:
        with sf.SoundFile(str(path)) as recording:
            if recording.frames == 0:
                raise STTError("The reference recording is empty.")
            if recording.frames > int(max_seconds * recording.samplerate):
                raise STTError(f"Use a complete recording of {max_seconds:g} seconds or less for transcription.")
            while True:
                samples = recording.read(recording.samplerate, dtype="float32", always_2d=True)
                if not samples.size:
                    break
                session.feed(recording.samplerate, samples)
        return session.finish()
    except STTError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise STTError(f"Could not read the reference recording: {exc}") from exc
    finally:
        session.close()

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from t2000 import stt


class FakeStream:
    def __init__(self):
        self.chunks, self.finish_count = [], 0

    def accept_waveform(self, sample_rate, values):
        if sample_rate != 16000:
            raise AssertionError("recognizer requires 16 kHz")
        self.chunks.append(np.array(values, copy=True))

    def input_finished(self):
        self.finish_count += 1


class FakeRecognizer:
    def __init__(self):
        self.streams = []

    def create_stream(self):
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        return False

    def get_result(self, stream):
        return "recognized reference"


class STTTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        stt._recognizers.clear()
        self.addCleanup(stt._recognizers.clear)

    def test_incremental_audio_consumed_once_and_finish_idempotent(self):
        recognizer = FakeRecognizer()
        with patch("t2000.stt._load_recognizer", return_value=recognizer):
            session = stt.STTSession(self.root)
            session.feed(16000, np.full(1600, 0.1, dtype=np.float32))
            session.feed(16000, np.full(800, 0.2, dtype=np.float32))
            self.assertEqual(session.finish(), "recognized reference")
            stream = recognizer.streams[0]
            total = sum(len(chunk) for chunk in stream.chunks)
            self.assertEqual(total, 8000 + 1600 + 800 + 12800)
            self.assertEqual(stream.finish_count, 1)
            session.finish()
            self.assertEqual(sum(len(chunk) for chunk in stream.chunks), total)
            with self.assertRaises(stt.STTError):
                session.feed(16000, np.zeros(10, dtype=np.float32))

    def test_native_rate_resampled_with_tail_flushed(self):
        recognizer = FakeRecognizer()
        with patch("t2000.stt._load_recognizer", return_value=recognizer):
            session = stt.STTSession(self.root)
            source = np.sin(np.arange(48000) * 0.07).astype(np.float32)
            session.feed(48000, source[:19001])
            session.feed(48000, source[19001:])
            session.finish()
            self.assertEqual(sum(len(chunk) for chunk in recognizer.streams[0].chunks), 8000 + 16000 + 12800)
            self.assertEqual(session.audio_seconds, 1)

    def test_recordings_have_separate_decoder_state(self):
        recognizer = FakeRecognizer()
        with patch("t2000.stt._load_recognizer", return_value=recognizer):
            first, second = stt.STTSession(self.root), stt.STTSession(self.root)
            first.feed(16000, np.ones(100, dtype=np.float32))
            second.feed(16000, np.ones(200, dtype=np.float32))
            self.assertIsNot(first._stream, second._stream)
            first.finish()
            self.assertFalse(second.finished)

    def test_bad_samples_changed_rate_and_oversize_rejected(self):
        recognizer = FakeRecognizer()
        with patch("t2000.stt._load_recognizer", return_value=recognizer):
            session = stt.STTSession(self.root, max_seconds=1)
            with self.assertRaises(stt.STTError):
                session.feed(16000, np.array([np.nan]))
            session.feed(16000, np.ones(100, dtype=np.float32))
            with self.assertRaises(stt.STTError):
                session.feed(48000, np.ones(100, dtype=np.float32))
            with self.assertRaises(stt.STTError):
                session.feed(16000, np.ones(16000, dtype=np.float32))
            self.assertEqual(session._input_samples, 100)
        with self.assertRaises(stt.STTError):
            stt.STTSession(self.root, num_threads=0)

    def test_missing_weights_and_optional_dependency_are_actionable(self):
        with self.assertRaisesRegex(stt.STTError, "fetch_stt.py"):
            stt._load_recognizer(self.root, 4)
        for name in stt.MODEL_FILES:
            (self.root / name).touch()
        with patch.dict(sys.modules, {"sherpa_onnx": None}):
            with self.assertRaisesRegex(stt.STTError, "optional STT"):
                stt._load_recognizer(self.root, 4)

    def test_recognizer_cache_respects_directory_and_threads(self):
        for name in stt.MODEL_FILES:
            (self.root / name).touch()
        calls = []
        def factory(**kwargs):
            calls.append(kwargs)
            return FakeRecognizer()
        module = types.SimpleNamespace(OnlineRecognizer=types.SimpleNamespace(from_transducer=factory))
        with patch.dict(sys.modules, {"sherpa_onnx": module}):
            first = stt._load_recognizer(self.root, 4)
            self.assertIs(first, stt._load_recognizer(self.root, 4))
            self.assertIsNot(first, stt._load_recognizer(self.root, 2))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["provider"], "cpu")

    def test_file_duration_rejected_before_loading_weights(self):
        source = self.root / "reference.wav"
        sf.write(source, np.ones(32000, dtype=np.float32), 16000)
        with patch("t2000.stt._load_recognizer") as loader:
            with self.assertRaisesRegex(stt.STTError, "complete"):
                stt.transcribe_file(source, self.root, max_seconds=1)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

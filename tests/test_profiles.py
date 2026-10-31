import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from t2000.audio import Audio, analyze_audio, load_audio
from t2000.errors import T2000Error
from t2000.profiles import ProfileStore


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "reference.wav"
        self.rate = 48000
        self.samples = (0.2 * np.sin(2 * np.pi * 200 * np.arange(self.rate) / self.rate)).astype(np.float32)
        sf.write(self.source, self.samples, self.rate, subtype="FLOAT")
        self.store = ProfileStore(self.root / "profiles")

    def test_native_rate_and_gain_preserved(self):
        stereo = np.column_stack((self.samples, self.samples * 0.5))
        sf.write(self.source, stereo, self.rate, subtype="FLOAT")
        audio = load_audio(self.source)
        self.assertEqual(audio.sample_rate, self.rate)
        np.testing.assert_allclose(audio.samples, self.samples * 0.75, atol=1e-8)
        self.assertEqual(audio.samples.dtype, np.float32)

    def test_invalid_and_oversized_audio_rejected_without_truncating(self):
        for invalid in (np.zeros(self.rate, dtype=np.float32),
                        np.full(self.rate, np.nan, dtype=np.float32),
                        self.samples[:100]):
            sf.write(self.source, invalid, self.rate, subtype="FLOAT")
            with self.assertRaises(T2000Error):
                load_audio(self.source)
        sf.write(self.source, self.samples, self.rate, subtype="FLOAT")
        with self.assertRaisesRegex(T2000Error, "complete"):
            load_audio(self.source, max_seconds=0.5, min_seconds=0.1)
        self.source.write_text("not an audio file")
        with self.assertRaisesRegex(T2000Error, "decode"):
            load_audio(self.source)

    def test_analysis_bounded_json_and_approximate_pitch(self):
        analysis = analyze_audio(load_audio(self.source))
        json.dumps(analysis, allow_nan=False)
        self.assertLessEqual(len(analysis["waveform"]["amplitude"]), 600)
        self.assertLessEqual(len(analysis["spectrum"]["frequency_hz"]), 256)
        self.assertLessEqual(len(analysis["pitch"]["f0_hz"]), 600)
        pitches = [v for v in analysis["pitch"]["f0_hz"] if v is not None]
        self.assertGreater(len(pitches), 20)
        self.assertAlmostEqual(float(np.median(pitches)), 200, delta=5)
        self.assertIn("Approximate", analysis["pitch"]["method"])
        self.assertEqual(analysis["clipped_fraction"], 0)
        with self.assertRaises(T2000Error):
            analyze_audio(Audio(np.array([np.nan], dtype=np.float32), self.rate))

    def test_same_content_reuses_immutable_artifact_and_name(self):
        first = self.store.create(self.source, "First name", "hello")
        reference = self.store.reference_path(first["id"])
        before = reference.read_bytes()
        second = self.store.create(self.source, "Replacement name", "different")
        self.assertEqual(first, second)
        self.assertEqual(first["sample_rate"], self.rate)
        self.assertEqual(sf.info(reference).subtype, "FLOAT")
        self.assertEqual(len(self.store.list()), 1)
        self.assertEqual(before, reference.read_bytes())

    def test_cache_key_tracks_transcript_backend_audio_and_preprocessing(self):
        first = self.store.create(self.source, "Voice", "hello")
        key = self.store.conditioning_dir(first["id"], "backend-v1")
        self.assertEqual(key, self.store.conditioning_dir(first["id"], "backend-v1"))
        self.assertNotEqual(key, self.store.conditioning_dir(first["id"], "backend-v2"))
        updated = self.store.update_transcript(first["id"], " corrected words ")
        self.assertEqual(updated["transcript"], "corrected words")
        self.assertNotEqual(key, self.store.conditioning_dir(first["id"], "backend-v1"))
        self.assertEqual(first["audio_sha256"], updated["audio_sha256"])
        with patch("t2000.profiles.PREPROCESSING_VERSION", "new-preprocessing-v2"):
            changed = self.store.create(self.source, "Changed preprocessing")
        self.assertNotEqual(first["id"], changed["id"])
        sf.write(self.source, self.samples * 0.8, self.rate, subtype="FLOAT")
        changed_audio = self.store.create(self.source, "Changed audio")
        self.assertNotEqual(first["id"], changed_audio["id"])

    def test_traversal_invalid_names_and_damaged_artifacts(self):
        for invalid in ("../outside", "a" * 63, "A" * 64, "", "a/b", None):
            with self.assertRaises(T2000Error):
                self.store.get(invalid)
        with self.assertRaises(T2000Error):
            self.store.create(self.source, " ")
        profile = self.store.create(self.source, "Voice")
        reference = self.store.reference_path(profile["id"])
        reference.write_bytes(b"corrupt")
        with self.assertRaisesRegex(T2000Error, "damaged"):
            self.store.get(profile["id"])

    def test_symlink_profiles_rejected(self):
        target = self.root / "outside"
        target.mkdir()
        (self.store.root / ("a" * 64)).symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(T2000Error, "symbolic"):
            self.store.get("a" * 64)

    def test_incomplete_metadata_rejected(self):
        profile = self.store.create(self.source, "Voice")
        del profile["transcript"]
        (self.store.root / profile["id"] / "metadata.json").write_text(json.dumps(profile))
        with self.assertRaisesRegex(T2000Error, "damaged"):
            self.store.get(profile["id"])

    def test_processed_voice_uses_new_conditioning_and_keeps_original(self):
        sf.write(self.source, np.tile(self.samples, 3), self.rate, subtype="FLOAT")
        original = self.store.create(self.source, "Original", "All words")
        original_path = self.store.reference_path(original["id"])
        before = original_path.read_bytes()
        edited = self.store.create(self.source, "Cropped", "Kept words", start_seconds=1, end_seconds=3,
                                   noise_reduction=True)
        self.assertNotEqual(original["id"], edited["id"])
        self.assertNotEqual(self.store.conditioning_dir(original["id"], "backend"),
                            self.store.conditioning_dir(edited["id"], "backend"))
        self.assertEqual(edited["duration_seconds"], 2)
        self.assertEqual(edited["sample_rate"], self.rate)
        self.assertEqual(original_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

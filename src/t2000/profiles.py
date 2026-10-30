"""Content-addressed reference artifacts and versioned conditioning cache paths."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import soundfile as sf

from .audio import PREPROCESSING_VERSION, analyze_audio, load_audio
from .errors import T2000Error


SCHEMA_VERSION = 1
_ID = re.compile(r"(?:[0-9a-f]{32}|[0-9a-f]{64})\Z")


class ProfileStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, profile_id: str) -> Path:
        if not isinstance(profile_id, str) or not _ID.fullmatch(profile_id):
            raise T2000Error("The reference profile ID is invalid.")
        directory = self.root / profile_id
        if directory.is_symlink():
            raise T2000Error("The reference profile directory must not be a symbolic link.")
        return directory

    def create(self, source_path: Path, name: str, transcript: str = "") -> dict:
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
            raise T2000Error("Use a reference name between 1 and 120 characters.")
        transcript = self._transcript(transcript)
        audio = load_audio(source_path)
        digest = hashlib.sha256()
        digest.update(f"{PREPROCESSING_VERSION}\n{audio.sample_rate}\n".encode())
        digest.update(np.asarray(audio.samples, dtype="<f4").tobytes())
        profile_id = digest.hexdigest()
        directory = self._directory(profile_id)
        if directory.exists():
            return self.get(profile_id)
        with tempfile.TemporaryDirectory(prefix=".profile-", dir=self.root) as temporary:
            staged = Path(temporary) / profile_id
            staged.mkdir()
            reference = staged / "reference.wav"
            sf.write(reference, audio.samples, audio.sample_rate, subtype="FLOAT")
            metadata = {
                "schema_version": SCHEMA_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
                "id": profile_id, "name": name.strip(), "transcript": transcript,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sample_rate": audio.sample_rate, "duration_seconds": audio.duration_seconds,
                "audio_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "analysis": analyze_audio(audio),
            }
            (staged / "metadata.json").write_text(json.dumps(metadata, allow_nan=False, indent=2) + "\n")
            try:
                staged.rename(directory)
            except OSError:
                if not directory.is_dir():
                    raise
                # Concurrent identical enrollment wins without rewriting its metadata.
                return self.get(profile_id)
        return metadata

    @staticmethod
    def _transcript(text: str) -> str:
        if not isinstance(text, str) or len(text) > 10000:
            raise T2000Error("Use a reference transcript of at most 10,000 characters.")
        return text.strip()

    def get(self, profile_id: str) -> dict:
        directory = self._directory(profile_id)
        metadata_path, reference = directory / "metadata.json", directory / "reference.wav"
        if metadata_path.is_symlink() or reference.is_symlink():
            raise T2000Error("Reference artifacts must not be symbolic links.")
        try:
            metadata = json.loads(metadata_path.read_text())
            if (not isinstance(metadata, dict) or metadata.get("id") != profile_id
                    or metadata.get("schema_version") != SCHEMA_VERSION
                    or not isinstance(metadata.get("preprocessing_version"), str)
                    or not isinstance(metadata.get("audio_sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", metadata["audio_sha256"])
                    or not all(isinstance(metadata.get(key), str) for key in ("name", "transcript", "created_at"))
                    or not isinstance(metadata.get("sample_rate"), int)
                    or not isinstance(metadata.get("duration_seconds"), (int, float))
                    or not isinstance(metadata.get("analysis"), dict)
                    or hashlib.sha256(reference.read_bytes()).hexdigest() != metadata["audio_sha256"]):
                raise T2000Error("The reference profile is damaged or has an unsupported schema. Enroll it again.")
            return metadata
        except FileNotFoundError as exc:
            raise T2000Error("This reference profile is unavailable. Enroll a recording first.") from exc
        except (OSError, ValueError, TypeError) as exc:
            raise T2000Error("The reference profile could not be read. Enroll the recording again.") from exc

    def list(self) -> list[dict]:
        profiles = [self.get(path.name) for path in self.root.iterdir()
                    if _ID.fullmatch(path.name) and path.is_dir()]
        return sorted(profiles, key=lambda value: value["created_at"], reverse=True)

    def reference_path(self, profile_id: str) -> Path:
        self.get(profile_id)
        return self._directory(profile_id) / "reference.wav"

    def conditioning_dir(self, profile_id: str, backend_signature: str | dict) -> Path:
        metadata = self.get(profile_id)
        try:
            key = hashlib.sha256(json.dumps({
                "schema_version": SCHEMA_VERSION, "preprocessing_version": metadata["preprocessing_version"],
                "audio_sha256": metadata["audio_sha256"], "transcript": metadata["transcript"],
                "backend_signature": backend_signature,
            }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        except (ValueError, TypeError) as exc:
            raise T2000Error("The synthesis backend supplied an invalid cache signature.") from exc
        directory = self.root / ".conditioning" / profile_id / key
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def update_transcript(self, profile_id: str, text: str) -> dict:
        metadata = self.get(profile_id)
        metadata["transcript"] = self._transcript(text)
        directory = self._directory(profile_id)
        fd, temporary = tempfile.mkstemp(prefix=".metadata-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w") as output:
                output.write(json.dumps(metadata, allow_nan=False, indent=2) + "\n")
            Path(temporary).replace(directory / "metadata.json")
        finally:
            Path(temporary).unlink(missing_ok=True)
        return metadata

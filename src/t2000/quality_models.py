"""Pinned CPU evaluation/VAD models; no implicit network access."""

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

from .errors import T2000Error


QUALITY_MODELS = {
    "silero_vad.onnx": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        "asset_id": 271935959,
        "size": 643854,
        "sha256": "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        "hash_source": "GitHub release asset digest, independently downloaded and verified",
    },
    "wespeaker_en_voxceleb_resnet34.onnx": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
               "speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx",
        "asset_id": 198883656,
        "size": 26534365,
        "sha256": "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94",
        "hash_source": "SHA-256 of complete official release asset; upstream digest unavailable",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path, spec):
    if path.stat().st_size != spec["size"]:
        raise ValueError("size mismatch")
    if sha256(path) != spec["sha256"]:
        raise ValueError("SHA-256 mismatch")


def verified_model_path(model_dir: Path, name: str) -> Path:
    """Return a pinned local model only after size and content verification."""
    if name not in QUALITY_MODELS:
        raise T2000Error(f"Unknown quality model: {name}")
    path = Path(model_dir).expanduser().resolve() / "quality" / name
    try:
        _verify(path, QUALITY_MODELS[name])
    except (OSError, ValueError) as exc:
        raise T2000Error(
            f"Quality model {name} is missing or invalid ({exc}). "
            f"Run python scripts/setup_quality_models.py --model-dir {Path(model_dir)}."
        ) from exc
    return path


def download_quality_models(model_dir: Path) -> list[Path]:
    """Explicit download with atomic publication; preserve prior files on failure."""
    root = Path(model_dir).expanduser().resolve() / "quality"
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, spec in QUALITY_MODELS.items():
        path = root / name
        try:
            _verify(path, spec)
        except (OSError, ValueError):
            fd, temporary = tempfile.mkstemp(prefix=f".{name}-", suffix=".part", dir=root)
            try:
                with os.fdopen(fd, "wb") as destination:
                    request = urllib.request.Request(spec["url"], headers={"User-Agent": "T-2000-model-setup"})
                    with urllib.request.urlopen(request, timeout=90) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            destination.write(block)
                _verify(Path(temporary), spec)
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        paths.append(path)
    return paths

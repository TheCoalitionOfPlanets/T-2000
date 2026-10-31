#!/usr/bin/env python3
"""Explicit pinned F5-TTS v1/Vocos preparation for a separate inference process.

No upstream install hooks or implicit ASR/UI/training dependencies are executed.
The only source change limits package exports to the two inference classes,
avoiding an eager Trainer import; network inference code is not used by worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request
import zipfile


SOURCE_REVISION = "45868abaabd74aabf03cd673d6dd00cc9d9c01ae"
MODEL_REVISION = "84e5a410d9cead4de2f847e7c9369a6440bdfaca"
VOCODER_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
SOURCE_FILE = f"downloads/F5-TTS-{SOURCE_REVISION}.zip"
INFERENCE_INIT = (
    "# T-2000 inference-only exports; model implementations are unchanged.\n"
    "from f5_tts.model.backbones.dit import DiT\n"
    "from f5_tts.model.cfm import CFM\n"
    "__all__ = [\"CFM\", \"DiT\"]\n"
).encode()
FILES = {
    SOURCE_FILE: {
        "url": f"https://codeload.github.com/SWivid/F5-TTS/zip/{SOURCE_REVISION}",
        "size": 1425291,
        "sha256": "4a5d4cd813784d3eb8ce7c5a06bbed3043093b3ea2fcd3b2cc73584fe09361d2",
    },
    "model.safetensors": {
        "url": f"https://huggingface.co/SWivid/F5-TTS/resolve/{MODEL_REVISION}/"
               "F5TTS_v1_Base/model_1250000.safetensors",
        "size": 1348435761,
        "sha256": "670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38",
    },
    "vocab.txt": {
        "url": f"https://huggingface.co/SWivid/F5-TTS/resolve/{MODEL_REVISION}/F5TTS_v1_Base/vocab.txt",
        "size": 13800,
        "git_blob": "cd934390e8f4b3ce98eb319ae618c084d01504b5",
    },
    "vocos/config.yaml": {
        "url": f"https://huggingface.co/charactr/vocos-mel-24khz/resolve/{VOCODER_REVISION}/config.yaml",
        "size": 461,
        "git_blob": "538262138a8b43863802f279909f26ec31c766b3",
    },
    "vocos/pytorch_model.bin": {
        "url": f"https://huggingface.co/charactr/vocos-mel-24khz/resolve/{VOCODER_REVISION}/pytorch_model.bin",
        "size": 54365991,
        "sha256": "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, spec: dict) -> None:
    if path.stat().st_size != spec["size"]:
        raise ValueError(f"Wrong file size: {path}")
    if "sha256" in spec and sha256(path) != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    if "git_blob" in spec:
        data = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != spec["git_blob"]:
            raise ValueError(f"Git blob mismatch: {path}")


def source_members(root: Path):
    """Yield verified archive bytes at their intended source-tree paths."""
    prefix = f"F5-TTS-{SOURCE_REVISION}/src/f5_tts/"
    with zipfile.ZipFile(root / SOURCE_FILE) as archive:
        for member in archive.infolist():
            if not member.filename.startswith(prefix) or member.is_dir():
                continue
            relative = PurePosixPath(member.filename[len(prefix):])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("Unsafe source archive entry")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Source archive symlinks are not supported")
            content = archive.read(member)
            if relative.as_posix() == "model/__init__.py":
                content = INFERENCE_INIT
            yield Path("sources/F5-TTS/src/f5_tts") / Path(*relative.parts), content


def check_bundle(root: Path) -> dict:
    root = Path(root).expanduser().resolve()
    for relative, spec in FILES.items():
        verify(root / relative, spec)
    sources = {}
    for relative, expected in source_members(root):
        path = root / relative
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError(f"F5 source integrity mismatch: {relative}")
        sources[relative.as_posix()] = hashlib.sha256(expected).hexdigest()
    required = "sources/F5-TTS/src/f5_tts/model/cfm.py"
    if required not in sources:
        raise ValueError("F5 source archive lacks the required model implementation")
    return {"schema": 1, "complete": True, "source_revision": SOURCE_REVISION,
            "model_revision": MODEL_REVISION, "vocoder_revision": VOCODER_REVISION,
            "model": "F5TTS_v1_Base", "sample_rate": 24000,
            "source_adjustment": "inference-only model/__init__.py exports; model kernels unchanged",
            "license": "F5 code MIT; pretrained F5 weights CC-BY-NC; Vocos MIT",
            "files": {relative: sha256(root / relative) for relative in FILES}, "sources": sources}


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for relative, spec in FILES.items():
        destination = root / relative
        try:
            verify(destination, spec)
            continue
        except (OSError, ValueError):
            pass
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".part", dir=destination.parent)
        try:
            print(f"Downloading {relative}", flush=True)
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "T-2000-F5-setup"})
            with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=90) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            verify(Path(temporary), spec)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    for relative, content in source_members(root):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    manifest = check_bundle(root)
    temporary = root / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(root / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/f5"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({"source_revision": SOURCE_REVISION, "files": FILES}, indent=2))
        return 0
    root = args.model_dir.expanduser().resolve()
    try:
        manifest = check_bundle(root) if args.check else prepare(root)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"F5 setup failed: {exc}\n")
    print(json.dumps({"verified": True, "model_dir": str(root),
                      "source_files": len(manifest["sources"]), "model": manifest["model"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

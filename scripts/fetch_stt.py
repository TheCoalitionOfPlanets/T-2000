#!/usr/bin/env python3
"""Explicitly fetch and verify pinned Nemotron INT8 weights for optional STT.

Adapted from https://github.com/TheCoalitionOfPlanets/T-1000/scripts/fetch_stt.py
No application request invokes this downloader.
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25"
ARCHIVE_NAME = MODEL_NAME + ".tar.bz2"
URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/" + ARCHIVE_NAME
# Pinned upstream release asset 405278878, inherited from T-1000's verified manifest.
SIZE = 463945051
SHA256 = "78e2b79fcf7271553a74402a76b771b09ea40117a39566a79f52235b23db6358"
FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def download(archive: Path):
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        partial = archive.with_suffix(archive.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > SIZE:
            partial.unlink()
            offset = 0
        if offset < SIZE:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("The server returned an unexpected download byte range.")
                else:
                    offset = 0
                print(f"Downloading {ARCHIVE_NAME} ({SIZE / 1e6:.0f} MB; resume at {offset / 1e6:.0f} MB)…", flush=True)
                with partial.open("ab" if offset else "wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
        if partial.stat().st_size != SIZE:
            raise RuntimeError("The model download is incomplete. Run this command again to resume.")
        partial.replace(archive)
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if archive.stat().st_size != SIZE or digest != SHA256:
        archive.unlink()
        raise RuntimeError("Model checksum verification failed. Cached archive removed; run again to retry.")


def install(model_dir: Path, archive: Path):
    marker = model_dir / ".archive-sha256"
    if marker.is_file() and marker.read_text().strip() == SHA256 and all((model_dir / f).is_file() for f in FILES):
        print(f"Speech recognition model is already installed: {model_dir}")
        return
    if model_dir.exists():
        raise RuntimeError(f"Unverified model directory already exists: {model_dir}. Move it aside first.")
    download(archive)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stt-download-", dir=model_dir.parent) as staging:
        extracted = Path(staging) / MODEL_NAME
        extracted.mkdir()
        with tarfile.open(archive, "r:bz2") as source:
            for filename in FILES:
                member = source.getmember(f"{MODEL_NAME}/{filename}")
                if not member.isfile():
                    raise RuntimeError(f"Unexpected model archive entry: {filename}")
                with source.extractfile(member) as input_file, (extracted / filename).open("wb") as output:
                    shutil.copyfileobj(input_file, output)
        (extracted / ".archive-sha256").write_text(SHA256 + "\n")
        extracted.rename(model_dir)
    print(f"Verified and installed speech recognition model: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data" / "models" / MODEL_NAME)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "stt")
    args = parser.parse_args()
    try:
        install(args.model_dir.expanduser(), args.cache_dir.expanduser() / ARCHIVE_NAME)
    except (OSError, ValueError, RuntimeError, tarfile.TarError, KeyError) as exc:
        parser.exit(1, f"Speech recognition download failed: {exc}\n")


if __name__ == "__main__":
    main()

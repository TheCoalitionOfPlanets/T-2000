#!/usr/bin/env python3
"""Fetch the pinned dots.tts SOAR/MF2 checkpoints into the local model directory.

Downloads only the files the worker loads, at the exact revisions the adapter
verifies, then checks the checkpoint SHA-256 against the same pins. Re-running is
cheap: complete model directories are left untouched.

The weights are Apache-2.0 from the dots-studio Hugging Face repositories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOS = {
    "soar": {"repo_id": "dots-studio/dots.tts-soar", "directory": "dots.tts-soar",
             "revision": "2f9b3e18d70d670d4c701da2dc55ded5755815ce",
             "sha256": "2787e6d4fe0b27ac33d28072abcadef53802c841bff037f3ee1f6bb5e1d3a2ce"},
    "mf2": {"repo_id": "dots-studio/dots.tts-mf-2steps", "directory": "dots.tts-mf-2steps",
            "revision": "159b33d33de0f9610d9ea73725a0820d27261fd7",
            "sha256": "5022f36ee599f037efc2869e351bfe2bb627573e05b45d5b1c99437bc674351a"},
}
MODEL_FILES = ("model.safetensors", "speaker_encoder.safetensors", "vocoder.safetensors", "config.json",
               "llm_config.json", "latent_stats.pt", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
               "chat_template.jinja")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete(directory: Path, pin: dict) -> bool:
    """A directory counts as ready only when every file exists and the checkpoint matches."""
    if not all((directory / name).is_file() for name in MODEL_FILES):
        return False
    return sha256(directory / "model.safetensors") == pin["sha256"]


def fetch(model: str, models_dir: Path, *, plan: bool) -> Path:
    pin = REPOS[model]
    destination = models_dir / pin["directory"]
    if complete(destination, pin):
        print(f"{model}: already present and verified at {destination}")
        return destination
    if plan:
        print(f"{model}: would download {pin['repo_id']}@{pin['revision'][:12]} "
              f"({len(MODEL_FILES)} files) -> {destination}")
        return destination

    from huggingface_hub import snapshot_download

    print(f"{model}: downloading {pin['repo_id']}@{pin['revision'][:12]} -> {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    # local_dir downloads straight into the model directory. Staging through the
    # shared hub cache first would hold a second full copy of every checkpoint.
    snapshot_download(
        repo_id=pin["repo_id"], revision=pin["revision"], local_dir=str(destination),
        allow_patterns=list(MODEL_FILES), max_workers=4,
    )
    for name in MODEL_FILES:
        if not (destination / name).is_file():
            raise SystemExit(f"{model}: {pin['repo_id']} is missing {name} at this revision.")
    actual = sha256(destination / "model.safetensors")
    if actual != pin["sha256"]:
        raise SystemExit(f"{model}: checkpoint SHA-256 {actual} does not match the pinned "
                         f"{pin['sha256']}. Refusing to use it.")
    print(f"{model}: verified {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"),
                        help="Directory that holds the dots.tts model directories.")
    parser.add_argument("--model", choices=(*REPOS, "all"), default="all")
    parser.add_argument("--plan", action="store_true", help="Report what would be fetched; no network use.")
    parser.add_argument("--check", action="store_true", help="Verify existing files and exit non-zero if incomplete.")
    args = parser.parse_args()

    models_dir = args.models_dir.expanduser().resolve()
    wanted = list(REPOS) if args.model == "all" else [args.model]
    if args.check:
        missing = [name for name in wanted if not complete(models_dir / REPOS[name]["directory"], REPOS[name])]
        print(json.dumps({"models_dir": str(models_dir), "ready": not missing, "missing": missing}, indent=2))
        return 1 if missing else 0
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        fetch(name, models_dir, plan=args.plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())

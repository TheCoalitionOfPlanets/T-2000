#!/usr/bin/env python3
"""Explicit, repeatable local model preparation; never invoked by the server.

Only downloads data and pinned source archives. It does not install packages or
execute upstream setup scripts. --check works offline with the standard library.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile


SOURCES = {
    "OpenVoice": ("myshell-ai/OpenVoice", "74a1d147b17a8c3092dd5430504bd83ef6c7eb23"),
    "MeloTTS": ("myshell-ai/MeloTTS", "209145371cff8fc3bd60d7be902ea69cbdb7965a"),
}
NLP_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
MODELS = {
    "myshell-ai/OpenVoiceV2": "f36e7edfe1684461a8343844af60babc2efbb727",
    "myshell-ai/MeloTTS-English": "bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef",
    "M4869/WavMark": "0ad3c7b74f641bddb61f6b85cdf2de0d93a5bfef",
}
# Upstream imports these tokenizers even for English. Only English BERT weights
# are downloaded. The alias is the exact name used by upstream from_pretrained.
FRONTENDS = {
    "google-bert/bert-base-uncased": ("bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
    "google-bert/bert-base-multilingual-uncased":
        ("bert-base-multilingual-uncased", "7cbf9a625e29989f6b9c6c2fa68234c304f7e38f"),
    "tohoku-nlp/bert-base-japanese-v3":
        ("tohoku-nlp/bert-base-japanese-v3", "65243d6e5629b969c77309f217bd7b1a79d43c7e"),
    "kykim/bert-kor-base": ("kykim/bert-kor-base", "1779cc0982ada0216dd6de0dd4e86fb78201926d"),
    "dbmdz/bert-base-french-europeana-cased":
        ("dbmdz/bert-base-french-europeana-cased", "b895c3cf291f7bf4c15639078a6bee0b3e272c5b"),
    "dccuchile/bert-base-spanish-wwm-uncased":
        ("dccuchile/bert-base-spanish-wwm-uncased", "d1c9c4565c9d6731e57ed7f027b802697bad861e"),
}
WEIGHT_HASHES = {
    "openvoice/converter/checkpoint.pth": "9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e",
    "melo/checkpoint.pth": "acd278040eaf9536908e2b965273df5a731c44d8f0da66cc5fed7972772ed23c",
    "watermark/model.pkl": "7a3d873d115c2c644685fb4cf109bdb299718296b88c871d3cf1c8e4cbc13144",
}
TOKENIZER_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "vocab.txt", "added_tokens.json"}
NLP_RESOURCES = {"corpora": ["cmudict"],
                 "taggers": ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"],
                 "tokenizers": ["punkt", "punkt_tab"]}
NLP_BLOBS = {
    "corpora/cmudict": ("48f1d64a431c474142969a793dfcffe3a2fc877b", 896069),
    "taggers/averaged_perceptron_tagger": ("d5bfb6852aa5d288c8f33b3f0ef0fde343afa405", 2526731),
    "taggers/averaged_perceptron_tagger_eng": ("b792e19546a5711218a05f4029ef17c8d62cd839", 1539115),
    "tokenizers/punkt": ("da7ffbd1e6fd6cc5c2f6879c2d4da23c7691944c", 13905355),
    "tokenizers/punkt_tab": ("5e5ff6137d5ee6025e400d1c3a7b21914c48b635", 4319076),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, *, expected_sha256=None, expected_git_blob=None, expected_size=None):
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"size mismatch: {path.name}")
    if expected_sha256 and sha256(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    if expected_git_blob:
        blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                blob.update(block)
        if blob.hexdigest() != expected_git_blob:
            raise ValueError(f"Git blob checksum mismatch: {path.name}")


def already_verified(path, **checks):
    if not path.is_file() or not (checks.get("expected_sha256") or checks.get("expected_git_blob")):
        return False
    try:
        verify_file(path, **checks)
        return True
    except (OSError, ValueError):
        return False


def download(url, target, *, expected_sha256=None, expected_git_blob=None, expected_size=None, timeout=120):
    target.parent.mkdir(parents=True, exist_ok=True)
    checks = {"expected_sha256": expected_sha256, "expected_git_blob": expected_git_blob,
              "expected_size": expected_size}
    if already_verified(target, **checks):
        return
    print(f"Downloading {target.name}", flush=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name, suffix=".part", delete=False) as out:
        temporary = Path(out.name)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "T-2000-setup/0.1"}),
                                        timeout=timeout) as response:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            out.flush()
            verify_file(temporary, **checks)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def github_blob(repo, revision, filename, target, *, expected_git_blob, expected_size):
    """Fetch immutable resource bytes through GitHub's API, then a raw fallback.

    The blob endpoint supports our ZIPs over 1 MB, for which the contents API
    omits content. Every route and every reused archive has the same pinned Git
    object checksum and size verified before it can be extracted.
    """
    checks = {"expected_git_blob": expected_git_blob, "expected_size": expected_size}
    if already_verified(target, **checks):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    endpoints = [f"https://api.github.com/repos/{repo}/git/blobs/{expected_git_blob}",
                 f"https://api.github.com/repos/{repo}/contents/{filename}?ref={revision}"]
    for endpoint in endpoints:
        print(f"Fetching pinned NLP resource {target.name} through GitHub API", flush=True)
        try:
            request = urllib.request.Request(endpoint, headers={"User-Agent": "T-2000-setup/0.1",
                                                                 "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(request, timeout=30) as source:
                result = json.load(source)
            if result.get("sha") != expected_git_blob or result.get("size") != expected_size:
                raise ValueError("GitHub resource metadata did not match the pinned blob")
            if result.get("encoding") != "base64":
                raise ValueError("GitHub endpoint omitted this resource's bytes")
            content = base64.b64decode("".join(result["content"].split()), validate=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as out:
                temporary = Path(out.name)
                try:
                    out.write(content)
                    out.flush()
                    verify_file(temporary, **checks)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            return
        except (OSError, ValueError, KeyError):
            # Some networks block the raw CDN, others block the API. All paths
            # remain pinned and checked, including after a transient API error.
            print(f"Trying another verified source for {target.name}", flush=True)
    download(f"https://github.com/{repo}/raw/{revision}/{filename}", target, timeout=30, **checks)


def extract(archive: Path, destination: Path, *, strip_first=False):
    """Extract regular files only; reject traversal and symlinks before writing."""
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                raise ValueError("unsafe archive path")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("archive contains a symbolic link")
            parts = path.parts[1:] if strip_first else path.parts
            if not parts:
                continue
            target = destination.joinpath(*parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("archive destination escapes model directory")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)


def model_metadata(repo, revision):
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as source:
        metadata = json.load(source)
    if metadata["sha"] != revision:
        raise ValueError(f"revision mismatch for {repo}")
    return {item["rfilename"]: item for item in metadata["siblings"]}


def hf_file(repo, revision, filename, destination, metadata, *, expected=None):
    item = metadata[filename]
    expected = expected or item.get("lfs", {}).get("sha256")
    download(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", destination,
             expected_sha256=expected, expected_git_blob=None if expected else item["blobId"],
             expected_size=item.get("size"))


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    # An interrupted setup must never advertise a complete set of models.
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest_path.rename(root / "manifest.previous.json")
    for name, (repo, revision) in SOURCES.items():
        archive = root / "downloads" / f"{name}-{revision}.zip"
        if not archive.exists():
            download(f"https://codeload.github.com/{repo}/zip/{revision}", archive)
        extract(archive, root / "sources" / name, strip_first=True)
    # Avoid an untracked source-directory pickle overriding the pinned CMU dict.
    english = root / "sources/MeloTTS/melo/text/english.py"
    original = english.read_text()
    if original.count("eng_dict = get_dict()") != 1:
        raise ValueError("pinned Melo English source did not match the documented cache patch")
    english.write_text(original.replace("eng_dict = get_dict()", "eng_dict = read_dict()"))
    for repo, revision in MODELS.items():
        metadata = model_metadata(repo, revision)
        if repo.endswith("OpenVoiceV2"):
            names = {"converter/config.json": "openvoice/converter/config.json",
                     "converter/checkpoint.pth": "openvoice/converter/checkpoint.pth"}
            names.update({name: f"openvoice/{name}" for name in metadata
                          if name.startswith("base_speakers/ses/en-") and name.endswith(".pth")})
        elif repo.endswith("MeloTTS-English"):
            names = {"config.json": "melo/config.json", "checkpoint.pth": "melo/checkpoint.pth"}
        else:
            name = "step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl"
            names = {name: "watermark/model.pkl"}
        for filename, relative in names.items():
            hf_file(repo, revision, filename, root / relative, metadata, expected=WEIGHT_HASHES.get(relative))
    for repo, (alias, revision) in FRONTENDS.items():
        metadata = model_metadata(repo, revision)
        names = sorted(TOKENIZER_FILES.intersection(metadata))
        if alias == "bert-base-uncased":
            names.append("model.safetensors")
        cache = root / "hf" / ("models--" + alias.replace("/", "--"))
        for filename in names:
            hf_file(repo, revision, filename, cache / "snapshots" / revision / filename, metadata)
        (cache / "refs").mkdir(parents=True, exist_ok=True)
        (cache / "refs/main").write_text(revision)
    # transformers 4.27.4 utils/hub.py reads integer 1 from this marker to
    # recognize the snapshots/refs layout and avoid an obsolete-cache migration.
    (root / "hf/version.txt").write_text("1")
    for category, names in NLP_RESOURCES.items():
        for name in names:
            archive = root / "downloads" / f"nltk-{name}-{NLP_REVISION}.zip"
            blob, size = NLP_BLOBS[f"{category}/{name}"]
            github_blob("nltk/nltk_data", NLP_REVISION, f"packages/{category}/{name}.zip", archive,
                        expected_git_blob=blob, expected_size=size)
            install_nlp_archive(archive, root / "nltk_data" / category, name)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
            relative = path.relative_to(root).as_posix()
            if relative.startswith(("sources/", "melo/", "openvoice/", "watermark/", "hf/", "nltk_data/")):
                files[relative] = sha256(path)
    manifest = {"schema": 1, "complete": True, "family": "melo-openvoice-v2",
                "sources": {name: {"repo": repo, "revision": revision,
                                   "archive_sha256": sha256(root / "downloads" / f"{name}-{revision}.zip")}
                            for name, (repo, revision) in SOURCES.items()},
                "model_revisions": MODELS,
                "frontend_revisions": {repo: revision for repo, (_, revision) in FRONTENDS.items()},
                "nltk_revision": NLP_REVISION,
                "nltk_git_blobs": NLP_BLOBS,
                "source_patch": "Melo English eng_dict uses read_dict rather than an untracked pickle cache",
                "files": files}
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return check(root)


def install_nlp_archive(archive, destination, name):
    # g2p_en 2.1.0 searches for these literal .zip paths during import, even
    # when the extracted resources exist. Keep both representations local.
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{name}.zip"
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".part", delete=False) as output:
        temporary = Path(output.name)
        try:
            with archive.open("rb") as source:
                shutil.copyfileobj(source, output)
            output.flush()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    extract(archive, destination)


def check(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != 1 or not manifest.get("complete") or not manifest.get("files"):
        raise ValueError("incomplete manifest")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"file checksum failed: {relative}")
    print(f"Verified {len(manifest['files'])} pinned local model/source/resource files in {root}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/models"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="verify the manifest offline; download nothing")
    group.add_argument("--plan", action="store_true", help="print pinned sources; download nothing")
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps({"source_revisions": SOURCES, "model_revisions": MODELS,
                          "frontends": FRONTENDS, "nltk_revision": NLP_REVISION,
                          "estimated_download": "About 0.9 GB including English BERT and pinned NLP resources; "
                                                "allow several GB free disk space including the Python environment."},
                         indent=2))
        return 0
    try:
        return check(args.model_dir.resolve()) if args.check else prepare(args.model_dir.resolve())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Model setup failed: {exc}. Fix the cause and rerun the explicit setup command.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Install a stub 'tn' package so dots.tts imports without WeTextProcessing.

WeTextProcessing depends on pynini, which has no Windows build: it passes GCC-only
flags to MSVC and fails to compile. dots_tts/utils/text.py imports the tn normalizers
at module scope, so the import must resolve even though this project never uses them --
the worker calls dots.tts with normalize_text=False, and the normalizers are built
lazily behind lru_cache, so nothing ever constructs one.

The stub therefore satisfies the import and raises only if a normalizer is actually
instantiated, which would mean text normalization got switched on and real
WeTextProcessing is needed.

Usage (Windows, where pynini cannot build):
  .venv\\Scripts\\python.exe scripts\\install_tn_stub.py
"""
from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

MESSAGE = ("WeTextProcessing is not installed. This is a stub provided by T-2000 because "
           "pynini has no Windows build. Text normalization is unavailable; the T-1000 "
           "worker does not use it (normalize_text=False). Install WeTextProcessing on a "
           "platform where pynini builds if you need normalization.")

BODY = f'''"""Stub normalizer -- see scripts/install_tn_stub.py."""


class Normalizer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError({MESSAGE!r})
'''


def main() -> int:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    try:
        import tn  # noqa: F401
    except ImportError:
        pass
    else:
        if not (site_packages / "tn" / ".t2000-stub").exists():
            print(f"tn is already installed at {Path(tn.__file__).parent} and is not our stub; "
                  f"leaving it alone.")
            return 0

    root = site_packages / "tn"
    for package in (root, root / "chinese", root / "english"):
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
    for language in ("chinese", "english"):
        (root / language / "normalizer.py").write_text(BODY)
    (root / ".t2000-stub").write_text("")

    print(f"Installed stub 'tn' package at {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

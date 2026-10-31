#!/usr/bin/env python3
"""Explicitly prepare the pinned offline Silero VAD and independent speaker scorer."""

import argparse
import json
from pathlib import Path

from t2000.config import Settings
from t2000.errors import T2000Error
from t2000.quality_models import QUALITY_MODELS, download_quality_models, verified_model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Settings.from_env().model_dir)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Print pinned resources without network or writes")
    mode.add_argument("--check", action="store_true", help="Verify existing resources offline")
    args = parser.parse_args()
    try:
        if args.plan:
            print(json.dumps(QUALITY_MODELS, indent=2))
            return 0
        paths = ([verified_model_path(args.model_dir, name) for name in QUALITY_MODELS]
                 if args.check else download_quality_models(args.model_dir))
        for path in paths:
            print(f"Verified {path}", flush=True)
    except (OSError, ValueError, T2000Error) as exc:
        parser.exit(1, f"Quality model setup failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

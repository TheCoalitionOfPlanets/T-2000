#!/usr/bin/env python3
"""Measure warm complete requests; enrollment/loading and browser playback are separate."""
import argparse
import csv
import json
import platform
import random
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from t2000.config import Settings
from t2000.doctor import inspect_runtime
from t2000.engine import Engine
from t2000.profiles import ProfileStore


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def summarize(records):
    complete = [record for record in records if record.get("complete")]
    result = {"requests": len(records), "completed": len(complete), "failed": len(records) - len(complete)}
    for key in ("generation_seconds", "first_pcm_seconds", "rtf", "audio_seconds",
                "base_seconds", "conversion_seconds", "max_phrase_gap_seconds"):
        values = [record["metrics"][key] for record in complete]
        result[key] = ({"p50": float(np.median(values)), "p95": float(np.percentile(values, 95))}
                       if values else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, action="append", help="Repeat for each voice")
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks/english.json")
    parser.add_argument("--output-dir", required=True, type=Path, help="Use a new directory for each run")
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--warmups", type=positive, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory to preserve earlier results")
    args.output_dir.mkdir(parents=True)
    settings = Settings.from_env()
    store = ProfileStore(settings.data_dir / "profiles")
    engine = Engine(settings, store)
    records = []
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "platform": platform.platform(), "runtime_before_load": inspect_runtime(settings),
              "timing_scope": "Warm worker start to final host PCM; includes all phrases. Excludes loading, "
                              "enrollment and browser playback; no browser first-audible measurement.",
              "limitations": ["No independent speaker verification or human ratings in this timing runner.",
                              "Short/long duration and speaking-rate changes require listening review.",
                              "CUDA allocated/reserved peaks exclude driver/other-process memory.",
                              "RSS is the process lifetime high-water mark, not a per-request peak.",
                              "p95 is descriptive; small request counts do not establish tail latency."],
              "setup": [], "warmups": [], "records": records}
    try:
        cases = json.loads(args.cases.read_text())
        if (not isinstance(cases, list) or not cases or any(
            not isinstance(case, dict) or not isinstance(case.get("id"), str)
            or not isinstance(case.get("text"), str) or not case["text"].strip() for case in cases
        )):
            raise ValueError("Cases must be a nonempty JSON array of {id, text} objects.")
        profiles = []
        for index, reference in enumerate(args.reference):
            started = time.perf_counter()
            profile = store.create(reference, f"Benchmark voice {index + 1}")
            setup = engine.prepare_profile(profile["id"])
            setup["enrollment_total_seconds"] = time.perf_counter() - started
            profiles.append(profile)
            report["setup"].append(setup)
        report["backend"] = engine.backend.info()
        report["signature"] = engine.backend.signature()
        import torch

        cuda_devices = sorted({torch.device(device).index or 0
                               for device in (settings.tts_device, settings.converter_device)
                               if device.startswith("cuda")}) if torch.cuda.is_available() else []
        report["cuda_devices"] = {
            str(device): {"name": torch.cuda.get_device_name(device),
                          "total_bytes": torch.cuda.get_device_properties(device).total_memory}
            for device in cuda_devices
        }
        for profile in profiles:
            for _ in range(args.warmups):
                report["warmups"].append(engine.generate(profile["id"], cases[0]["text"], seed=args.seed))
        requests = [(profile, case, repeat) for profile in profiles for case in cases
                    for repeat in range(1, args.repeats + 1)]
        random.Random(args.seed).shuffle(requests)
        for index, (profile, case, repeat) in enumerate(requests):
            record = {"profile_id": profile["id"], "case": case["id"], "repeat": repeat,
                      "text": case["text"], "seed": args.seed}
            try:
                for device in cuda_devices:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                result = engine.generate(profile["id"], case["text"], seed=args.seed)
                record.update(result)
                record["complete"] = True
                record["wav_path"] = str((settings.data_dir / "output" / f"{result['job_id']}.wav").resolve())
                record["cuda_memory"] = {}
                for device in cuda_devices:
                    torch.cuda.synchronize(device)
                    record["cuda_memory"][str(device)] = {
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    }
                record["process_rss_high_water_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            except Exception as exc:
                record.update(complete=False, error=f"{type(exc).__name__}: {exc}")
            records.append(record)
            report["summary"] = summarize(records)
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"{index + 1}/{len(requests)} {case['id']} repeat {repeat}: "
                  f"{'complete' if record['complete'] else record['error']}", flush=True)
    except Exception as exc:
        report["setup_error"] = f"{type(exc).__name__}: {exc}"
        print(f"Benchmark stopped: {report['setup_error']}", file=sys.stderr)
    finally:
        report["summary"] = summarize(records)
        report["runtime_after"] = inspect_runtime(settings)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        with (args.output_dir / "requests.csv").open("w", newline="") as output:
            columns = ["case", "profile_id", "repeat", "complete", "generation_seconds", "first_pcm_seconds",
                       "audio_seconds", "rtf", "base_seconds", "conversion_seconds", "wav_path", "error"]
            writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow(dict(record, **record.get("metrics", {})))
        summary = report["summary"]
        text = ["# T-2000 timing run", "", report["timing_scope"], "",
                f"Completed {summary['completed']} / {summary['requests']} timed requests.", "",
                "| Measurement | p50 | p95 |", "|---|---:|---:|"]
        for key, value in summary.items():
            if isinstance(value, dict):
                text.append(f"| {key} | {value['p50']:.4f} | {value['p95']:.4f} |")
        if "setup_error" in report:
            text.extend(["", "Setup failed: " + report["setup_error"]])
        text.extend(["", "## Limits", "", *[f"- {item}" for item in report["limitations"]], ""])
        (args.output_dir / "REPORT.md").write_text("\n".join(text))
    if report.get("setup_error") or report["summary"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

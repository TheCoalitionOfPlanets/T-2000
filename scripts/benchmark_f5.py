#!/usr/bin/env python3
"""Compare isolated F5 direct cloning against an existing matched Melo control."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import select
import subprocess

from compare_voice_quality import BASELINE, checkpoint, score_outputs, write_products
from t2000.config import Settings
from t2000.profiles import ProfileStore
from t2000.quality import audio_metrics
from t2000.stt import transcribe_file


def exchange(process, request, timeout=600):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError("F5 worker response timed out")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"F5 worker exited with code {process.poll()}")
    response = json.loads(line)
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "F5 worker returned an error"))
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, default=Path("data/benchmarks/quality-first/report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/f5-first"))
    parser.add_argument("--python", type=Path, default=Path(".venv-f5/bin/python"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/f5"))
    parser.add_argument("--steps", type=int, choices=range(8, 65), default=32)
    parser.add_argument("--seed", type=int, help="Override candidate seed; missing control pairs remain excluded")
    parser.add_argument("--only-profile", help="Unique prefix of profile ID for a diagnostic subset")
    parser.add_argument("--reference-preprocessing", choices=("raw", "upstream-edges"), default="upstream-edges")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory to preserve prior results.")
    args.output_dir.mkdir(parents=True)
    baseline = json.loads(args.baseline_report.read_text())
    controls = [row for row in baseline["records"] if row["variant"] == BASELINE]
    if args.only_profile:
        controls = [row for row in controls if row["profile_id"].startswith(args.only_profile)]
    if not controls:
        parser.error("No matched baseline rows found.")
    settings = Settings.from_env()
    variant = f"F5TTS_v1_Base/{args.steps}steps/{args.reference_preprocessing}/reference-pace"
    store = ProfileStore(settings.data_dir / "profiles")
    report = {key: copy.deepcopy(baseline[key]) for key in
              ("schema", "limitations", "references", "cases", "seeds", "quality_models", "scorer", "asr")
              if key in baseline}
    report.update(created_at=datetime.now(timezone.utc).isoformat(), baseline=BASELINE,
                  variants=[{"id": BASELINE}, {"id": variant}], records=copy.deepcopy(controls), setup=[],
                  baseline_report=str(args.baseline_report.resolve()), warmup="One excluded F5 warmup; control reused.")
    reference_texts = {}
    for reference in report["references"]:
        profile = store.get(reference["profile_id"])
        text = profile["transcript"].strip()
        source = "saved_profile_transcript"
        if not text:
            text = reference.get("recognized_text", "").strip()
            source = "baseline_reference_asr"
        if not text:
            text = transcribe_file(Path(reference["reference_path"]), settings.stt_model_dir,
                                   num_threads=settings.num_threads, max_seconds=30)
            source = "nemotron_reference_asr"
        if not text.strip():
            raise ValueError("F5 needs a nonempty reference transcript; edit this saved voice first.")
        reference.update(f5_reference_text=text, f5_reference_text_source=source)
        reference_texts[reference["reference_sha256"]] = (text, source)
    process = None
    try:
        with (args.output_dir / "worker.log").open("w") as log:
            process = subprocess.Popen(
                [str(args.python.absolute()), str(Path(__file__).with_name("f5_infer_worker.py")),
                 "--model-dir", str(args.model_dir.resolve()), "--device", "cuda:0"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
            )
            loaded = exchange(process, {"command": "load"})
            report["setup"].append(loaded)
            print("F5 weights loaded on GPU", flush=True)
            first = controls[0]
            request = {"reference_path": first["reference_path"],
                       "reference_text": reference_texts[first["reference_sha256"]][0],
                       "text": first["text"], "seed": args.seed if args.seed is not None else first["seed"],
                       "speed": 1.0, "steps": args.steps, "reference_preprocessing": args.reference_preprocessing,
                       "output_path": str((args.output_dir / "warmup.wav").resolve())}
            report["warmup_result"] = exchange(process, request)
            print("F5 warmup completed", flush=True)
            for index, control in enumerate(controls):
                row = {key: copy.deepcopy(control[key]) for key in
                       ("reference_sha256", "reference_path", "profile_id", "case", "text", "expected_text", "seed")}
                text, source = reference_texts[row["reference_sha256"]]
                if args.seed is not None:
                    row["seed"] = args.seed
                row.update(variant=variant, complete=False, score_status="pending", backend_sample_rate=24000,
                           reference_text=text, reference_text_source=source)
                try:
                    response = exchange(process, {
                        "reference_path": row["reference_path"], "reference_text": text, "text": row["text"],
                        "seed": row["seed"], "speed": 1.0, "steps": args.steps,
                        "reference_preprocessing": args.reference_preprocessing,
                        "output_path": str((args.output_dir / f"f5-{index:02d}-{row['case']}.wav").resolve()),
                    })
                    row.update(complete=True, wav_path=response["wav_path"], worker=response,
                               metrics={key: response[key] for key in
                                        ("generation_seconds", "first_pcm_seconds", "audio_seconds", "rtf",
                                         "clipped_fraction", "peak_before_clip", "request_seconds")},
                               cuda_memory=response["cuda_memory"], audio=audio_metrics(response["wav_path"]))
                except Exception as exc:
                    row["generation_error"] = f"{type(exc).__name__}: {exc}"
                report["records"].append(row)
                checkpoint(report, args.output_dir)
                print(f"F5 {index + 1}/{len(controls)} {row['case']}: {row['complete']}", flush=True)
            exchange(process, {"command": "close"}, timeout=30)
            process.wait(timeout=30)
            process = None
        # Independent scoring starts only after the GPU worker has exited.
        score_outputs(report, settings, args.output_dir)
    except Exception as exc:
        report["setup_error"] = f"{type(exc).__name__}: {exc}"
        print(report["setup_error"], flush=True)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        write_products(report, args.output_dir)
    print(json.dumps({"summaries": report.get("summaries"), "paired_summary": report.get("paired_summary")}, indent=2))
    return int(bool(report.get("setup_error") or any(
        not row["complete"] or row.get("score_status") != "ok" for row in report["records"]
    )))


if __name__ == "__main__":
    raise SystemExit(main())

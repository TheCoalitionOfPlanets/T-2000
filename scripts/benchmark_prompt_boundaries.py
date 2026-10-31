#!/usr/bin/env python3
"""Test punctuation and quiet reference edges without editing the original voice."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import soxr

import compare_voice_quality as reporting
from t2000.backends.t1000 import T1000Backend
from t2000.config import Settings
from t2000.profiles import ProfileStore
from t2000.quality import audio_metrics
from t2000.quality_models import sha256, verified_model_path


BASELINE = "T1000/MF2/native-delivery"


def speech_edge_branch(path, model_dir, padding_ms=150):
    """Keep all detected speech and native samples; cut only demonstrably quiet outer edges."""
    import sherpa_onnx

    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    detection = soxr.resample(mono, rate, 16000, quality="HQ") if rate != 16000 else mono
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(verified_model_path(model_dir, "silero_vad.onnx"))
    config.sample_rate = 16000
    config.silero_vad.threshold = .5
    config.silero_vad.min_silence_duration = .2
    config.silero_vad.min_speech_duration = .1
    detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
    regions = []

    def drain():
        while not detector.empty():
            segment = detector.front
            regions.append((segment.start / 16000, (segment.start + len(segment.samples)) / 16000))
            detector.pop()

    for index in range(0, len(detection), 512):
        detector.accept_waveform(detection[index:index + 512])
        drain()
    detector.flush()
    drain()
    if not regions:
        raise ValueError("No speech detected; do not trim this reference.")
    start = max(0, round((regions[0][0] - padding_ms / 1000) * rate))
    end = min(len(mono), round((regions[-1][1] + padding_ms / 1000) * rate))
    active_start, active_end = round(regions[0][0] * rate), round(regions[-1][1] * rate)

    def rms(values):
        return float(np.sqrt(np.mean(values.astype(np.float64) ** 2))) if values.size else 0.0

    active_rms = rms(mono[active_start:active_end])
    leading_rms, trailing_rms = rms(mono[:start]), rms(mono[end:])
    if active_rms <= 0 or max(leading_rms, trailing_rms) > active_rms * .3:
        raise ValueError("Excluded outer edges are not clearly quiet; preserve the full reference.")
    if not 1 <= (end - start) / rate <= 6:
        raise ValueError("Derived reference is outside the complete 1–6 second comparison bound.")
    return mono[start:end], rate, {
        "detector": "verified Silero ONNX CPU,16k,threshold0.5,min_silence0.2,min_speech0.1",
        "speech_regions_seconds": regions, "padding_ms": padding_ms,
        "selected_native_frames": [start, end], "selected_seconds": [start / rate, end / rate],
        "active_rms": active_rms, "excluded_leading_rms": leading_rms,
        "excluded_trailing_rms": trailing_rms,
        "quiet_gate": "Excluded edge RMS must not exceed 0.3 times detected speech RMS",
        "operation": "Only external edges removed; internal pauses and original sample values retained",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1000-root", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--baseline-report", type=Path, default=Path("data/benchmarks/delivery-own-first/report.json"))
    parser.add_argument("--conditions", nargs="+", choices=("period-only", "quiet-edges"),
                        default=["period-only", "quiet-edges"])
    parser.add_argument("--candidate-label", help="Explicit label for a single reviewed-transcript condition")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a fresh output directory.")
    if args.candidate_label and len(args.conditions) != 1:
        parser.error("--candidate-label requires exactly one condition.")
    args.output_dir.mkdir(parents=True)
    settings = Settings.from_env()
    store = ProfileStore(settings.data_dir / "profiles")
    profile = store.get(args.profile)
    path = store.reference_path(args.profile)
    original_hash = sha256(path)
    original_text = profile["transcript"].strip()
    if not original_text:
        parser.error("The exact saved reference transcript is required; no nickname guessing is allowed.")
    punctuated_text = original_text if original_text.endswith((".", "!", "?", "。")) else original_text + "."
    previous = json.loads(args.baseline_report.read_text())
    controls = [copy.deepcopy(row) for row in previous["records"]
                if row["variant"] == BASELINE and row["profile_id"] == args.profile]
    if len(controls) != 2 or any(row["seed"] != 42 for row in controls):
        parser.error("This diagnostic requires the two original MF2 seed42 control requests.")
    reporting.BASELINE = BASELINE
    condition_labels = {"period-only": "MF2/period-only", "quiet-edges": "MF2/period-plus-vad-quiet-edges"}
    if args.candidate_label:
        condition_labels[args.conditions[0]] = args.candidate_label
    conditions = list(dict.fromkeys(args.conditions))
    variants = [condition_labels[condition] for condition in conditions]
    report = {"schema": 1, "created_at": datetime.now(timezone.utc).isoformat(), "baseline": BASELINE,
              "references": [{"profile_id": args.profile, "reference_path": str(path),
                              "reference_sha256": original_hash, "reference_text": original_text}],
              "variants": [{"id": BASELINE}, *[{"id": variant} for variant in variants]],
              "records": copy.deepcopy(controls), "setup": [], "seeds": [42],
              "cases": [{"id": row["case"], "text": row["text"]} for row in controls],
              "limitations": [*reporting.LIMITATIONS,
                              "Each condition records its transcript and optional external-edge changes explicitly.",
                              "The nickname remains the saved user transcript; no spelling is guessed by this diagnostic.",
                              "All similarity scoring uses the original full reference, not the shortened branch.",
                              "Check prompt-like inserted words before accepting any similarity change."],
              "input_provenance": {"original_reference_sha256": original_hash,
                                   "original_transcript": original_text, "candidate_transcript": punctuated_text,
                                   "baseline_reference_texts": sorted({row.get("reference_text", "")
                                                                        for row in controls}),
                                   "transcript_differs_from_baseline": any(
                                       row.get("reference_text", "") != punctuated_text for row in controls),
                                   "conditions": conditions,
                                   "baseline_report": str(args.baseline_report.resolve()),
                                   "baseline_report_sha256": sha256(args.baseline_report)}}
    backend = None
    try:
        input_paths = {"period-only": path}
        if "quiet-edges" in conditions:
            branch, rate, details = speech_edge_branch(path, settings.model_dir)
            branch_path = args.output_dir / "conditioning-quiet-edges.wav"
            sf.write(branch_path, branch, rate, subtype="FLOAT")
            report["input_provenance"]["edge_candidate"] = {
                **details, "path": str(branch_path.resolve()), "sha256": sha256(branch_path),
                "audio": audio_metrics(branch_path),
            }
            input_paths["quiet-edges"] = branch_path
        backend = T1000Backend(settings.model_dir, python_executable=args.t1000_root / ".venv/bin/python",
                               t1000_root=args.t1000_root, model="mf2", device=settings.tts_device,
                               num_threads=settings.num_threads, reference_boundary="raw")
        backend.load()
        for condition in conditions:
            variant, input_path = condition_labels[condition], input_paths[condition]
            cache = args.output_dir / ("conditioning-" + condition)
            backend.prepare(input_path, cache, reference_text=punctuated_text)
            report["setup"].append({"variant": variant, "input_path": str(input_path.resolve()),
                                     "input_sha256": sha256(input_path), "reference_text": punctuated_text,
                                     "backend": backend.info()})
            # Exclude one warmup for each distinct conditioning candidate.
            backend.synthesize(controls[0]["text"], cache, speed=1, seed=42)
            for index, control in enumerate(tuple(controls)):
                row = {key: control[key] for key in ("profile_id", "reference_path", "reference_sha256",
                                                    "case", "text", "expected_text", "seed")}
                row.update(variant=variant, complete=False, score_status="pending",
                           conditioning_reference_path=str(input_path.resolve()),
                           conditioning_reference_sha256=sha256(input_path), reference_text=punctuated_text)
                try:
                    started = time.perf_counter()
                    chunk = backend.synthesize(row["text"], cache, speed=1, seed=42)
                    elapsed = time.perf_counter() - started
                    output = args.output_dir / f"{condition}-{index}.wav"
                    sf.write(output, chunk.samples, chunk.sample_rate, subtype="PCM_16")
                    row.update(complete=True, wav_path=str(output.resolve()), audio=audio_metrics(output),
                               backend_sample_rate=chunk.sample_rate, worker=backend.info()["last_worker"],
                               metrics={"generation_seconds": chunk.base_seconds,
                                        "first_pcm_seconds": elapsed, "request_seconds": elapsed,
                                        "audio_seconds": len(chunk.samples) / chunk.sample_rate,
                                        "rtf": chunk.base_seconds / (len(chunk.samples) / chunk.sample_rate),
                                        "clipped_fraction": 0.0})
                except Exception as exc:
                    row["generation_error"] = f"{type(exc).__name__}: {exc}"
                report["records"].append(row)
                reporting.checkpoint(report, args.output_dir)
                print(f"{variant} {row['case']}: {row['complete']}", flush=True)
        backend.close()
        backend = None
        reporting.score_outputs(report, settings, args.output_dir)
    except Exception as exc:
        report["setup_error"] = f"{type(exc).__name__}: {exc}"
        print(report["setup_error"], flush=True)
    finally:
        if backend is not None:
            backend.close()
        if sha256(path) != original_hash or store.get(args.profile)["transcript"] != profile["transcript"]:
            report["original_changed_error"] = "Original profile changed while diagnostic ran; do not compare silently."
        reporting.write_products(report, args.output_dir)
    return int(bool(report.get("setup_error") or report.get("original_changed_error")
                    or any(not row["complete"] or row["score_status"] != "ok" for row in report["records"])))


if __name__ == "__main__":
    raise SystemExit(main())

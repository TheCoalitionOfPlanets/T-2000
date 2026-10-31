#!/usr/bin/env python3
"""Compare the user's own reference in F5 and the installed T-1000 quality controls."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import html
from pathlib import Path
import time

import compare_voice_quality as reporting
from t2000.backends.f5 import F5Backend
from t2000.backends.t1000 import T1000Backend
from t2000.config import Settings
from t2000.engine import Engine
from t2000.profiles import ProfileStore
from t2000.quality import audio_metrics
from t2000.quality_models import sha256
from t2000.stt import transcribe_file


F5_LABEL = "F5/32steps/upstream-edges/reference-pace"
DEFAULT_CASES = [
    {"id": "natural-statement", "text": "I was going to call you earlier, but the day just got away from me."},
    {"id": "natural-question", "text": "Wait, are you serious? I thought we were meeting tomorrow, not today."},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1000-root", required=True, type=Path)
    parser.add_argument("--t1000-python", type=Path, help="Defaults to T-1000's existing .venv/bin/python")
    parser.add_argument("--f5-python", type=Path)
    parser.add_argument("--profile", action="append", help="Repeat saved profile IDs; defaults to latest profile")
    parser.add_argument("--reference", action="append", type=Path, help="Enroll an explicit complete reference")
    parser.add_argument("--reference-text", help="Exact transcript for one reference/profile; no automatic rewrite")
    parser.add_argument("--cases", type=Path, help="Cases JSON array; defaults to two natural sentences")
    parser.add_argument("--models", nargs="+", choices=("f5", "soar", "mf2"), default=["f5", "soar", "mf2"])
    parser.add_argument("--reference-boundary", choices=("sentence", "raw"), default="sentence",
                        help="Sentence adds neutral terminal punctuation; raw reproduces the original control")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a fresh output directory to preserve earlier comparisons.")
    if not 0 <= args.seed < 2**32 or args.warmups < 0:
        parser.error("Seed must be in [0,4294967295] and warmups must be nonnegative.")
    args.output_dir.mkdir(parents=True)
    settings = Settings.from_env()
    store = ProfileStore(settings.data_dir / "profiles")
    # Pair against current F5, not against a public voice or an uncalibrated score threshold.
    reporting.BASELINE = F5_LABEL
    models = list(dict.fromkeys(["f5", *args.models]))
    labels = {"f5": F5_LABEL, "soar": "T1000/SOAR10/native-delivery", "mf2": "T1000/MF2/native-delivery"}
    if args.reference_boundary == "sentence":
        labels.update(soar="T1000/SOAR10/sentence-reference-boundary", mf2="T1000/MF2/sentence-reference-boundary")
    report = {"schema": 1, "created_at": datetime.now(timezone.utc).isoformat(), "baseline": F5_LABEL,
              "variants": [{"id": labels[model], "model": model} for model in models],
              "references": [], "records": [], "setup": [], "seeds": [args.seed],
              "limitations": [*reporting.LIMITATIONS,
                              "The listening target is the user's reference. T-1000 is a working control, "
                              "not the target speaker or a presumed winner.",
                              "Accent, sentence flow, naturalness and emotion require reference-guided listening.",
                              "F5, SOAR and MF2 run sequentially. Native output rates are retained; "
                              "shared evaluation resamples independent scoring inputs only.",
                              "No post-hoc pitch, tempo or emotion modification is applied. "
                              "Multiplicative peak attenuation, if needed, is recorded."],
              "listening_questions": ["Which version sounds like you?", "Does the accent match your reference?",
                                      "Does the sentence flow naturally?", "Does the emotion fit the sentence?",
                                      "Are pauses, pace and emphasis convincing?"],
              "t1000_root": str(args.t1000_root.expanduser().resolve()),
              "t1000_reference_boundary": args.reference_boundary}
    try:
        cases = reporting.read_cases(args.cases) if args.cases else DEFAULT_CASES
        report["cases"] = cases
        if args.cases:
            report["cases_sha256"] = sha256(args.cases)
        profiles = [store.get(profile) for profile in args.profile or []]
        for path in args.reference or []:
            profiles.append(store.create(path, "Delivery comparison reference", args.reference_text or ""))
        if not profiles:
            profiles = store.list()[:1]
        if not profiles:
            raise ValueError("No saved reference exists; pass --profile or --reference.")
        if args.reference_text and len(profiles) != 1:
            raise ValueError("--reference-text requires exactly one reference/profile.")
        seen = set()
        for profile in profiles:
            if profile["id"] in seen:
                continue
            seen.add(profile["id"])
            path = store.reference_path(profile["id"])
            audio = audio_metrics(path)
            if not audio["valid"] or not 1 <= audio["duration_seconds"] <= 6:
                raise ValueError("Use the same complete 1–6 second own-voice reference for every control.")
            text = args.reference_text or profile["transcript"].strip()
            text_source = "explicit_reviewed_text" if args.reference_text else "saved_profile_transcript"
            if not text:
                text = transcribe_file(path, settings.stt_model_dir,
                                       num_threads=settings.num_threads, max_seconds=6)
                text_source = "nemotron_reference_asr_review_needed"
            if not text.strip():
                raise ValueError("No reference words recognized. Supply the exact reference transcript.")
            report["references"].append({"profile_id": profile["id"], "reference_path": str(path),
                                         "reference_sha256": sha256(path), "reference_audio": audio,
                                         "reference_text": text, "reference_text_source": text_source})
        for variant in report["variants"]:
            model = variant["model"]
            if model == "f5":
                backend = F5Backend(settings.model_dir, python_executable=args.f5_python or settings.f5_python,
                                    device=settings.tts_device, num_threads=settings.num_threads, steps=32)
            else:
                backend = T1000Backend(
                    settings.model_dir,
                    python_executable=args.t1000_python or args.t1000_root / ".venv/bin/python",
                    t1000_root=args.t1000_root, model=model, device=settings.tts_device,
                    num_threads=settings.num_threads, reference_boundary=args.reference_boundary,
                )
            engine = Engine(replace(settings, backend="f5-tts-v1" if model == "f5" else f"t1000-{model}"),
                            store, backend)
            try:
                for reference in report["references"]:
                    # The descriptor includes the exact same text for all candidates.
                    setup = {"variant": variant["id"], "reference_sha256": reference["reference_sha256"]}
                    failed_setup = None
                    try:
                        started = time.perf_counter()
                        backend.load()
                        signature = backend.signature()
                        cache = store.conditioning_dir(reference["profile_id"], {
                            "backend": signature, "comparison_transcript": reference["reference_text"],
                        })
                        backend.prepare(Path(reference["reference_path"]), cache,
                                        reference_text=reference["reference_text"])
                        setup.update(signature=signature, preparation_total_seconds=time.perf_counter() - started,
                                     backend=backend.info())
                        for warmup in range(args.warmups):
                            backend.synthesize(cases[0]["text"], cache, speed=1.0, seed=args.seed)
                    except Exception as exc:
                        failed_setup = f"{type(exc).__name__}: {exc}"
                        setup["error"] = failed_setup
                    report["setup"].append(setup)
                    for index, case in enumerate(cases):
                        row = {**reference, "variant": variant["id"], "case": case["id"], "text": case["text"],
                               "expected_text": case.get("spoken_expected_text", case["text"]), "seed": args.seed,
                               "complete": False, "score_status": "pending"}
                        try:
                            if failed_setup:
                                raise RuntimeError(failed_setup)
                            started = time.perf_counter()
                            chunk = backend.synthesize(case["text"], cache, speed=1.0, seed=args.seed)
                            elapsed = time.perf_counter() - started
                            import soundfile as sf

                            path = args.output_dir / f"{model}-{reference['profile_id'][:12]}-{index:02d}.wav"
                            sf.write(path, chunk.samples, chunk.sample_rate, subtype="PCM_16")
                            row.update(complete=True, wav_path=str(path.resolve()), audio=audio_metrics(path),
                                       backend_sample_rate=chunk.sample_rate, worker=backend.info().get("last_worker"),
                                       metrics={"generation_seconds": chunk.base_seconds + chunk.conversion_seconds,
                                                "first_pcm_seconds": elapsed,
                                                "audio_seconds": len(chunk.samples) / chunk.sample_rate,
                                                "rtf": (chunk.base_seconds + chunk.conversion_seconds)
                                                       / (len(chunk.samples) / chunk.sample_rate),
                                                "request_seconds": elapsed, "clipped_fraction": 0.0})
                        except Exception as exc:
                            row["generation_error"] = f"{type(exc).__name__}: {exc}"
                        report["records"].append(row)
                        reporting.checkpoint(report, args.output_dir)
                        print(f"{variant['id']} {case['id']}: {row['complete']}", flush=True)
            finally:
                engine.close()
        reporting.score_outputs(report, settings, args.output_dir)
    except Exception as exc:
        report["setup_error"] = f"{type(exc).__name__}: {exc}"
        print(report["setup_error"], flush=True)
    finally:
        reporting.write_products(report, args.output_dir)
        listening = args.output_dir / "LISTEN.html"
        instructions = ('<h1>Your reference voice and delivery</h1><p>The target is your own reference. '
                        'T-1000 is a working comparison, not the target speaker.</p><ul>'
                        + ''.join(f'<li>{html.escape(question)}</li>' for question in report["listening_questions"])
                        + '</ul>')
        listening.write_text(listening.read_text().replace('<h1>Matched voice listening</h1>', instructions))
    return int(bool(report.get("setup_error") or any(not row["complete"] or row["score_status"] != "ok"
                                                     for row in report["records"])))


if __name__ == "__main__":
    raise SystemExit(main())

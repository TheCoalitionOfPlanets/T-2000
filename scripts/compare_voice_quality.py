#!/usr/bin/env python3
"""Generate matched voice candidates, then score and build an offline listening report."""

import argparse
import base64
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import html
import itertools
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from t2000.backends.melo_openvoice import MeloOpenVoiceBackend
from t2000.config import Settings
from t2000.engine import Engine
from t2000.profiles import ProfileStore
from t2000.quality import (SpeakerScorer, audio_metrics, descriptive_features, error_metrics,
                          paired_comparisons, summarize)
from t2000.quality_models import QUALITY_MODELS, sha256, verified_model_path
from t2000.stt import MODEL_FILES, transcribe_file


BASELINE = "EN-US/full/tau=0.3"
LIMITATIONS = [
    "Independent speaker cosine is an objective proxy, not a likeness percentage or a calibrated pass threshold.",
    "ASR WER/CER measure one recognizer's agreement, not human-rated intelligibility or naturalness.",
    "No automatic winner or default promotion. Review paired audio, transcripts, failures and duration changes.",
    "Reference conditioning and output scoring are separate. Scoring uses full mono clips with HQ 16 kHz resampling.",
    "Generation metrics exclude model loading, reference preparation and subsequent CPU scoring; "
    "they do not measure browser audible latency.",
    "A small set from one reference cannot establish quality for other speakers, accents or recording conditions.",
    "Pitch, spectrum and speaking pace are descriptive diagnostics, not identity or smoothness quality scores.",
]


def label(speaker, mode, tau):
    return f"{speaker}/{mode}/tau={tau:g}"


def read_cases(path):
    cases = json.loads(path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty array of {id, text} objects.")
    ids = set()
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip()
                or not isinstance(case.get("text"), str) or not case["text"].strip()
                or case["id"] in ids):
            raise ValueError("Cases require unique nonempty string IDs and text.")
        expected = case.get("spoken_expected_text", case["text"])
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError("Expected transcripts must be nonempty text.")
        error_metrics(expected, expected)
        ids.add(case["id"])
    return cases


def refresh_summaries(report):
    rows = report["records"]
    report["summaries"] = {variant["id"]: summarize([row for row in rows if row["variant"] == variant["id"]])
                           for variant in report["variants"]}
    report["per_reference"] = {
        reference["reference_sha256"]: {
            variant["id"]: summarize([row for row in rows if row["variant"] == variant["id"]
                                      and row["reference_sha256"] == reference["reference_sha256"]])
            for variant in report["variants"]}
        for reference in report["references"]
    }
    report["pairs"] = paired_comparisons(rows, BASELINE)
    paired_summary = {}
    for variant in report["variants"]:
        if variant["id"] == BASELINE:
            continue
        pairs = [pair for pair in report["pairs"] if pair["variant"] == variant["id"]]
        complete = [pair for pair in pairs if pair["eligible"]]
        paired_summary[variant["id"]] = {
            "pairs": len(pairs), "eligible": len(complete), "excluded": len(pairs) - len(complete),
            "median_speaker_cosine_delta": statistics.median(
                pair["speaker_cosine_delta"] for pair in complete) if complete else None,
            "total_word_error_delta": sum(pair["word_error_delta"] for pair in complete),
            "duration_changes_over_ten_percent": sum(pair["duration_change_over_ten_percent"]
                                                      for pair in complete),
            "assessment": "No automatic winner; review each pair and listening report.",
        }
    report["paired_summary"] = paired_summary


def checkpoint(report, output_dir):
    refresh_summaries(report)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def audio_tag(path):
    try:
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f'<audio controls preload="none" src="data:audio/wav;base64,{encoded}"></audio>'
    except OSError as exc:
        return html.escape(f"Audio unavailable: {exc}")


def write_products(report, output_dir):
    checkpoint(report, output_dir)
    columns = ["variant", "reference_sha256", "case", "seed", "complete", "score_status",
               "speaker_cosine", "wer", "cer", "generation_seconds", "first_pcm_seconds", "rtf",
               "audio_seconds", "sample_rate", "backend_sample_rate", "frames", "clipped_samples",
               "pitch_median_hz", "pitch_iqr_hz",
               "words_per_second_including_pauses", "reference_words_per_second_including_pauses",
               "wav_path", "generation_error", "score_error"]
    with (output_dir / "requests.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in report["records"]:
            flat = {**row, **row.get("metrics", {}), **row.get("audio", {})}
            flat.update(wer=row.get("asr", {}).get("word", {}).get("rate"),
                        cer=row.get("asr", {}).get("character", {}).get("rate"),
                        pitch_median_hz=row.get("descriptive", {}).get("pitch", {}).get("median_hz"),
                        pitch_iqr_hz=row.get("descriptive", {}).get("pitch", {}).get("iqr_hz"),
                        words_per_second_including_pauses=row.get("descriptive", {}).get("pace", {}).get(
                            "words_per_second_including_pauses"),
                        reference_words_per_second_including_pauses=(row.get("reference_descriptive") or {}).get(
                            "pace", {}).get("words_per_second_including_pauses"))
            writer.writerow(flat)
    lines = ["# Matched voice experiment", "", *[f"- {text}" for text in LIMITATIONS], "",
             "| Candidate | Generated / requests | Scored | Eligible | Median speaker cosine | WER |",
             "|---|---:|---:|---:|---:|---:|"]
    for variant, summary in report["summaries"].items():
        cosine = summary["speaker_cosine"]
        cosine_text = f"{cosine['median']:.4f}" if cosine else "—"
        wer = summary["word"]["rate"]
        wer_text = f"{wer:.4f}" if wer is not None else "—"
        lines.append(f"| {variant} | {summary['generated']} / {summary['requests']} | {summary['scored']} | "
                     f"{summary['eligible']} | {cosine_text} | {wer_text} |")
    if report.get("setup_error"):
        lines += ["", "Setup error: " + report["setup_error"]]
    if report.get("scoring_error"):
        lines += ["", "Scoring error: " + report["scoring_error"]]
    lines += ["", "Open [the self-contained listening page](LISTEN.html) for reference and candidate audio.",
              "Detailed pairs, sample counts, failures, hashes and model pins are in `report.json`.", ""]
    (output_dir / "REPORT.md").write_text("\n".join(lines))
    page = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>T-2000 matched voice listening</title><style>',
            'body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#faf9f6}',
            'section{border-top:1px solid #bbb;padding:18px 0}article{margin:14px 0;padding:12px;background:white}',
            'audio{display:block;width:min(100%,650px)}pre{white-space:pre-wrap}small{color:#444}</style>',
            '<h1>Matched voice listening</h1><p>No human quality ratings have been assigned.</p><ul>',
            *[f'<li>{html.escape(text)}</li>' for text in LIMITATIONS], '</ul>']
    grouped = {}
    for row in report["records"]:
        grouped.setdefault((row["reference_sha256"], row["case"], row["seed"]), []).append(row)
    for (reference, case, seed), rows in sorted(grouped.items()):
        page += [f'<section><h2>{html.escape(case)} · seed {seed}</h2>',
                 f'<p>{html.escape(rows[0]["text"])}</p>',
                 f'<small>Reference SHA-256: {html.escape(reference)}</small>',
                 '<h3>Reference</h3>', audio_tag(rows[0]["reference_path"])]
        for row in sorted(rows, key=lambda value: (value["variant"] != BASELINE, value["variant"])):
            score = row.get("speaker_cosine")
            page += [f'<article><h3>{html.escape(row["variant"])}</h3>']
            if row.get("wav_path"):
                page.append(audio_tag(row["wav_path"]))
            detail = {key: row.get(key) for key in ("recognized_text", "generation_error", "score_error")}
            detail.update(speaker_cosine=score, audio=row.get("audio"), asr=row.get("asr"),
                          descriptive=row.get("descriptive"), reference_descriptive=row.get("reference_descriptive"))
            page += [f'<pre>{html.escape(json.dumps(detail, indent=2))}</pre></article>']
        page.append('</section>')
    page.append('</html>')
    (output_dir / "LISTEN.html").write_text("\n".join(page))


def release_models(engine):
    """Drop each candidate's weights, including Melo's module-global English BERT."""
    engine._backend = None
    module = sys.modules.get("melo.text.english_bert")
    if module is not None:
        module.model = None
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_variant(variant, settings, store, references, cases, seeds, warmups, report, output_dir):
    setting_fields = {"speaker": variant["speaker"]}
    # Root settings may expose these for the studio; the injected backend always receives them.
    if hasattr(settings, "reference_mode"):
        setting_fields["reference_mode"] = variant["reference_mode"]
    if hasattr(settings, "conversion_tau"):
        setting_fields["conversion_tau"] = variant["tau"]
    candidate_settings = replace(settings, **setting_fields)
    engine = Engine(candidate_settings, store, backend=MeloOpenVoiceBackend(
        settings.model_dir, tts_device=settings.tts_device, converter_device=settings.converter_device,
        speaker=variant["speaker"], language=settings.language, num_threads=settings.num_threads,
        reference_mode=variant["reference_mode"], tau=variant["tau"],
    ))
    setup_errors = {}
    try:
        for reference in references:
            setup = {"variant": variant["id"], "reference_sha256": reference["reference_sha256"]}
            try:
                setup.update(engine.prepare_profile(reference["profile_id"]))
                for _ in range(warmups):
                    engine.generate(reference["profile_id"], cases[0]["text"], seed=seeds[0])
            except Exception as exc:
                setup["error"] = f"{type(exc).__name__}: {exc}"
                setup_errors[reference["profile_id"]] = setup["error"]
            report["setup"].append(setup)
        variant["backend"] = engine.backend.info()
        requests = list(itertools.product(references, cases, seeds))
        random.Random(42).shuffle(requests)
        for reference, case, seed in requests:
            row = dict(reference, variant=variant["id"], case=case["id"], text=case["text"],
                       expected_text=case.get("spoken_expected_text", case["text"]), seed=seed,
                       backend_sample_rate=variant.get("backend", {}).get("sample_rate"),
                       complete=False, score_status="pending")
            try:
                if reference["profile_id"] in setup_errors:
                    raise RuntimeError(setup_errors[reference["profile_id"]])
                result = engine.generate(reference["profile_id"], case["text"], seed=seed)
                row.update(result, complete=True)
                wav = settings.data_dir / "output" / f"{result['job_id']}.wav"
                row.update(wav_path=str(wav.resolve()), audio=audio_metrics(wav))
            except Exception as exc:
                row["generation_error"] = f"{type(exc).__name__}: {exc}"
            report["records"].append(row)
            checkpoint(report, output_dir)
            print(f"Generated {variant['id']} {case['id']} seed {seed}: "
                  f"{'complete' if row['complete'] else row['generation_error']}", flush=True)
    finally:
        release_models(engine)


def score_outputs(report, settings, output_dir):
    scorer = SpeakerScorer(settings.model_dir, num_threads=min(4, settings.num_threads))
    cache = {}
    reference_features = {}
    for reference in report["references"]:
        try:
            path = Path(reference["reference_path"])
            digest = sha256(path)
            if digest not in cache:
                cache[digest] = transcribe_file(path, settings.stt_model_dir,
                                               num_threads=settings.num_threads, max_seconds=30)
            reference["recognized_text"] = cache[digest]
            reference["descriptive"] = descriptive_features(path, cache[digest])
            reference_features[reference["reference_sha256"]] = reference["descriptive"]
        except Exception as exc:
            reference["scoring_error"] = f"{type(exc).__name__}: {exc}"
    for row in report["records"]:
        if not row.get("complete"):
            row.update(score_status="error", score_error="No complete generated output.")
            continue
        try:
            if not row.get("audio", {}).get("valid"):
                raise ValueError("Generated waveform is empty, silent, non-finite or not mono.")
            wav = Path(row["wav_path"])
            digest = row["audio"]["wav_sha256"]
            started = time.perf_counter()
            if digest not in cache:
                cache[digest] = transcribe_file(wav, settings.stt_model_dir,
                                               num_threads=settings.num_threads, max_seconds=180)
            row["recognized_text"] = cache[digest]
            row["asr"] = error_metrics(row["expected_text"], row["recognized_text"])
            row["speaker_cosine"] = scorer.cosine(Path(row["reference_path"]), wav)
            row["descriptive"] = descriptive_features(wav, row["recognized_text"])
            row["reference_descriptive"] = reference_features.get(row["reference_sha256"])
            row.update(score_status="ok", scoring_seconds=time.perf_counter() - started)
        except Exception as exc:
            row.update(score_status="error", score_error=f"{type(exc).__name__}: {exc}")
        checkpoint(report, output_dir)
        print(f"Scored {row['variant']} {row['case']} seed {row['seed']}: {row['score_status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", type=Path,
                        help="Repeat for each reference; defaults to every enrolled profile")
    parser.add_argument("--all-profiles", action="store_true", help="Also include every enrolled reference")
    parser.add_argument("--cases", type=Path,
                        default=Path(__file__).resolve().parents[1] / "benchmarks/english.json")
    parser.add_argument("--speakers", nargs="+", default=["EN-US"],
                        choices=["EN-US", "EN-BR", "EN-INDIA", "EN-AU", "EN-DEFAULT"])
    parser.add_argument("--reference-modes", nargs="+", default=["full", "speech"], choices=["full", "speech"])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a fresh directory to preserve previous experiments")
    if (any(not 0 <= tau <= 1 for tau in args.taus) or any(not 0 <= seed <= 0xFFFFFFFF for seed in args.seeds)
            or args.warmups < 0):
        parser.error("taus must be finite in [0,1], seeds in [0,4294967295], warmups nonnegative")
    args.output_dir.mkdir(parents=True)
    settings = Settings.from_env()
    variants = {BASELINE: {"id": BASELINE, "speaker": "EN-US", "reference_mode": "full", "tau": 0.3}}
    for speaker, mode, tau in itertools.product(args.speakers, args.reference_modes, args.taus):
        key = label(speaker, mode, tau)
        variants[key] = {"id": key, "speaker": speaker, "reference_mode": mode, "tau": tau}
    report = {
        "schema": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "settings": asdict(settings),
        "baseline": BASELINE, "variants": list(variants.values()), "references": [],
        "cases_path": str(args.cases.resolve()), "seeds": list(dict.fromkeys(args.seeds)),
        "warmups_per_reference_and_variant": args.warmups, "quality_models": QUALITY_MODELS,
        "limitations": LIMITATIONS, "setup": [], "records": [],
        "source_urls": [
            "https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/"
            "openvoice/se_extractor.py",
            "https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/"
            "openvoice/models.py",
            "https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speaker-identification.py",
        ],
        "scorer": {"provider": "cpu", "model": "WeSpeaker English VoxCeleb ResNet34 ONNX",
                   "independent_of_synthesis_conditioning": True,
                   "preprocessing": "Full mono clip, soxr HQ to 16000 Hz, no crops or silence removal, L2 embedding"},
    }
    report["settings"] = {key: str(value) if isinstance(value, Path) else value
                          for key, value in report["settings"].items()}
    try:
        cases = read_cases(args.cases)
        report.update(cases=cases, cases_sha256=sha256(args.cases))
        for name in QUALITY_MODELS:
            verified_model_path(settings.model_dir, name)
        manifest_path = settings.model_dir / "manifest.json"
        report["synthesis_manifest"] = {"sha256": sha256(manifest_path),
                                        "contents": json.loads(manifest_path.read_text())}
        report["asr"] = {"provider": "cpu", "model_dir": str(settings.stt_model_dir),
                         "files_sha256": {name: sha256(settings.stt_model_dir / name) for name in MODEL_FILES}}
        store = ProfileStore(settings.data_dir / "profiles")
        paths = list(args.reference or [])
        if args.all_profiles or not paths:
            paths += [store.reference_path(profile["id"]) for profile in store.list()]
        hashes = set()
        for index, path in enumerate(paths):
            digest = sha256(path)
            if digest in hashes:
                continue
            hashes.add(digest)
            profile = store.create(path, f"Quality reference {index + 1}")
            reference_path = store.reference_path(profile["id"])
            report["references"].append({"source_path": str(path.resolve()), "reference_sha256": digest,
                                         "profile_id": profile["id"], "reference_path": str(reference_path),
                                         "enrolled_reference_sha256": sha256(reference_path),
                                         "reference_audio": audio_metrics(reference_path)})
        if not report["references"]:
            raise ValueError("No references supplied or enrolled. Pass --reference /path/to/reference.wav.")
        for variant in report["variants"]:
            generate_variant(variant, settings, store, report["references"], cases, report["seeds"],
                             args.warmups, report, args.output_dir)
        try:
            score_outputs(report, settings, args.output_dir)
        except Exception as exc:
            report["scoring_error"] = f"{type(exc).__name__}: {exc}"
            for row in report["records"]:
                if row["score_status"] == "pending":
                    row.update(score_status="error", score_error=report["scoring_error"])
    except Exception as exc:
        report["setup_error"] = f"{type(exc).__name__}: {exc}"
        print(report["setup_error"], file=sys.stderr, flush=True)
    finally:
        write_products(report, args.output_dir)
    print(f"Reports: {args.output_dir.resolve()}", flush=True)
    return int(bool(report.get("setup_error") or report.get("scoring_error")
                    or any(not row["complete"] or row["score_status"] != "ok" for row in report["records"])))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Synthesize a labeled WAV dataset for one saved voice, to bootstrap voice-conversion
(e.g. RVC) training data when real recordings of that voice are not yet available.

Reference-conditioned synthesis (T-1000 SOAR/MF2, F5, Melo/OpenVoice) is used to
generate many distinct sentences in the voice, each saved as its own clip with its
transcript recorded alongside it. This is bootstrap data, not a substitute for real
recordings: quality is capped by the synthesis backend's own fidelity, and any of its
artifacts will be baked into whatever is trained on this output.

An existing dataset in the output directory is extended rather than replaced: its clips
and transcripts are kept, --minutes is the total target across old and new clips, and
sentences already recorded are never synthesized twice. Pass --fresh to start over.

Usage:
  .venv/bin/python scripts/generate_voice_dataset.py jack
  .venv/bin/python scripts/generate_voice_dataset.py jack --minutes 30 --backend t1000-mf2
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import json
import random
import re
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2000.config import Settings  # noqa: E402
from t2000.engine import Engine  # noqa: E402
from t2000.errors import T2000Error  # noqa: E402
from t2000.profiles import ProfileStore  # noqa: E402
from t2000.solo import resolve_profile  # noqa: E402

BACKENDS = ("t1000-soar", "t1000-mf2", "f5-tts-v1", "melo-openvoice-v2")

# Hand-written, original, phonetically and prosodically varied sentences: a mix of
# statements, questions and exclamations across everyday topics. Shuffled first, so
# the earliest (highest-value) minutes of the dataset carry the most natural variety.
BASE_SENTENCES = [
    "The morning started quietly, with just the sound of rain on the window.",
    "Could you pass me that folder before the meeting starts?",
    "Honestly, I didn't expect the bridge to be this long.",
    "She set the kettle down and stared out at the garden for a while.",
    "Why does the printer always jam right before a deadline?",
    "We walked along the shore until the tide came in.",
    "That's the third time this week the bus has been late.",
    "He kept the letter in his coat pocket for years before reading it.",
    "Turn left at the old bakery, then it's the second house on the right.",
    "I can't believe how quickly the semester went by.",
    "The dog barked twice, then settled back down by the fire.",
    "Are you sure this is the right trailhead?",
    "A thin mist hung over the valley until well past noon.",
    "Nobody warned me the elevator would be out of service today.",
    "She hummed an old tune while sorting through the mail.",
    "What time does the market close on Sundays?",
    "The engineer double-checked every bolt before signing off.",
    "It was colder than expected, so we stayed inside and played cards.",
    "He laughed so hard he nearly dropped the tray.",
    "Remember to water the plants before you leave for the weekend.",
    "The orchestra tuned their instruments as the lights dimmed.",
    "I think the recipe calls for twice as much flour.",
    "You should have seen the look on her face when the lights came on.",
    "The old clock in the hallway still keeps perfect time.",
    "Where did you leave the keys last night?",
    "A sudden gust knocked the umbrella right out of his hand.",
    "The report is due Friday, but I'd like it sooner if possible.",
    "She practiced the same four bars until her fingers ached.",
    "Watch your step, the floor is still wet from mopping.",
    "They argued for a while, then simply agreed to disagree.",
    "The harbor was quiet except for the gulls circling overhead.",
    "I wonder if the train still stops at that little station.",
    "He measured the shelf twice before cutting the wood.",
    "The children raced each other to the end of the pier.",
    "Isn't it strange how familiar this street still feels?",
    "The candles flickered as the door swung open.",
    "We should probably leave before the traffic gets worse.",
    "Her handwriting was small but surprisingly easy to read.",
    "The storm passed just as quickly as it had arrived.",
    "Can you believe they're renovating the whole library?",
    "He whistled the same three notes every time he walked past.",
    "The coffee had gone cold by the time she remembered it.",
    "Set the oven to medium and check it again in ten minutes.",
    "I never did find out who left that note on my desk.",
    "The hikers paused at the ridge to catch their breath.",
    "Didn't you say the shop was closing early today?",
    "A single lamp lit the whole room from the corner.",
    "The mechanic shook his head and reached for another wrench.",
    "We spent the whole afternoon arguing about nothing important.",
    "The river was higher than usual after last week's rain.",
    "Please make sure the back door is locked before you go.",
    "She read the letter twice, then folded it carefully away.",
    "That old radio still works, believe it or not.",
    "The crowd fell silent the moment the curtain rose.",
    "I'd rather walk than wait another twenty minutes for the bus.",
    "He sketched the outline first, then filled in the color.",
    "The orchard smelled of apples long before we reached it.",
    "Did anyone actually confirm the reservation for tonight?",
    "The porch light flickered twice before finally going out.",
    "She tied her shoes, grabbed her bag, and was out the door.",
    "It's hard to explain, but the whole place felt familiar.",
    "The committee will announce its decision sometime next week.",
    "He never did learn to fold a map the right way.",
    "The waves kept pulling the sandcastle apart before we finished it.",
    "Could you check if the attic light is still working?",
    "The bakery on the corner always smells like cinnamon in the morning.",
    "I thought the meeting was moved to Thursday, not Wednesday.",
    "The old dog barely lifted its head when the doorbell rang.",
    "They repainted the fence a shade lighter than before.",
    "Honestly, I'd forgotten how loud this street gets at night.",
    "The last train leaves the platform at exactly midnight.",
    "She balanced the tray carefully as she crossed the kitchen.",
    "Why does this drawer always stick in the summer?",
    "The lecture ran long, so we skipped the usual break.",
    "A faint smell of smoke drifted in from somewhere down the road.",
    "He counted the coins twice before handing them over.",
    "The garden looked wild after just two weeks of neglect.",
    "Isn't the conference supposed to start at nine sharp?",
    "She whispered the answer so only I could hear it.",
    "The ferry rocked gently as it pulled away from the dock.",
    "I keep meaning to fix that hinge, but I never get around to it.",
]

# Word banks for an open-ended templated filler, used once the hand-written list is
# exhausted, so a run can target any length without ever needing the same sentence twice.
_SUBJECTS = ["the engineer", "my neighbor", "the old captain", "a quiet student", "the shopkeeper",
            "her brother", "the night guard", "an impatient driver", "the young violinist",
            "our new tenant", "the retired teacher", "a curious child", "the ferry operator"]
_VERBS = ["noticed", "mentioned", "forgot about", "carefully repaired", "quietly questioned",
         "suddenly remembered", "patiently explained", "accidentally dropped", "slowly uncovered",
         "briefly described", "nervously approached", "finally finished"]
_OBJECTS = ["the broken gate", "an old family photograph", "the misplaced ledger", "a tangled fishing line",
           "the half-written letter", "a faded road sign", "the rusted bicycle", "an unopened package",
           "the missing puzzle piece", "a chipped porcelain cup", "the overgrown hedge", "a borrowed umbrella"]
_WHEN = ["just after sunrise", "right before the storm", "during the long drive home", "near the end of autumn",
        "while everyone else slept", "on the quietest day of the week", "just as the bell rang",
        "long after the guests had left", "somewhere past midnight", "in the middle of the argument"]
_TAGS = ["and nobody seemed to mind.", "though it took far longer than expected.", "without saying a word.",
        "which surprised almost everyone.", "and then simply walked away.", "but thought better of it.",
        "and quietly smiled.", "before anyone could ask why."]


def templated_filler(rng: random.Random):
    """An unbounded generator of fresh, grammatically varied sentences."""
    while True:
        sentence = (f"{rng.choice(_SUBJECTS).capitalize()} {rng.choice(_VERBS)} "
                   f"{rng.choice(_OBJECTS)} {rng.choice(_WHEN)}, {rng.choice(_TAGS)}")
        yield sentence


def corpus(rng: random.Random):
    """Shuffled hand-written sentences first, then endless templated filler."""
    shuffled = list(BASE_SENTENCES)
    rng.shuffle(shuffled)
    yield from shuffled
    yield from templated_filler(rng)


def slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return value or "voice"


def load_existing(output_dir: Path, clips_dir: Path) -> list[dict]:
    """Clips already on disk, read back from transcripts.tsv so a run can extend them."""
    transcripts = output_dir / "transcripts.tsv"
    if not transcripts.exists():
        return []
    records = []
    for line in transcripts.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        file, duration, text = line.split("\t", 2)
        if (clips_dir / file).exists():
            records.append({"file": file, "text": text, "duration_seconds": float(duration)})
    return records


def write_metadata(output_dir: Path, profile: dict, args, records: list[dict], total_seconds: float) -> None:
    (output_dir / "transcripts.tsv").write_text(
        "file\tduration_seconds\ttext\n" + "".join(
            f"{row['file']}\t{row['duration_seconds']:.3f}\t{row['text']}\n" for row in records))
    (output_dir / "manifest.json").write_text(json.dumps({
        "voice_name": profile["name"], "profile_id": profile["id"], "backend": args.backend,
        "device": args.device, "seed": args.seed, "clip_count": len(records),
        "total_duration_seconds": total_seconds, "target_minutes": args.minutes,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Bootstrap data synthesized from a reference-conditioned backend, not real recordings. "
               "Quality is capped by that backend's own fidelity.",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", help="Saved voice name or ID (for example: jack)")
    parser.add_argument("--minutes", type=float, default=120.0, help="Target dataset duration in minutes")
    parser.add_argument("--backend", choices=BACKENDS, default="t1000-soar", help="Synthesis backend to use")
    parser.add_argument("--device", default="cuda:0", help="Device for the synthesis backend")
    parser.add_argument("--output-dir", type=Path, help="Defaults to data/voice_datasets/<voice>")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle and synthesis seed, for reproducibility")
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="Abort if this many sentences in a row fail to synthesize")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing clips and generate the whole dataset from scratch")
    parser.add_argument("--workers", type=int, default=1,
                        help="Synthesis processes to run concurrently. Each holds its own copy of "
                             "the model on the GPU, so the ceiling is VRAM: at batch size 1 a single "
                             "stream leaves the GPU mostly idle, and concurrency is what fills it.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use at least one worker.")

    settings = replace(Settings.from_env(), backend=args.backend, tts_device=args.device)
    store = ProfileStore(settings.data_dir / "profiles")
    engine = Engine(settings, store)

    profile = resolve_profile(store, args.profile)
    available = {item["id"]: item["available"] for item in engine.available_backends()}
    if not available.get(args.backend):
        raise SystemExit(f"T-2000: {args.backend} is not available on this system "
                         f"(device={args.device}). Run 't2000 doctor' to check requirements.")
    if profile["duration_seconds"] > engine.max_reference_seconds:
        raise SystemExit(f"T-2000: {profile['name']!r}'s reference is {profile['duration_seconds']:.1f}s, "
                         f"over the {engine.max_reference_seconds}s limit for {args.backend}.")

    output_dir = args.output_dir or (settings.data_dir / "voice_datasets" / slug(profile["name"]))
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    records = [] if args.fresh else load_existing(output_dir, clips_dir)
    already_done = {row["text"] for row in records}
    total_seconds = sum(row["duration_seconds"] for row in records)
    if records:
        print(f"T-2000: found {len(records)} existing clips ({total_seconds / 60:.2f} min) in {output_dir}; "
              f"extending to {args.minutes:g} minutes.")

    target_seconds = args.minutes * 60
    if total_seconds >= target_seconds:
        print(f"T-2000: already at {total_seconds / 60:.2f} minutes, at or over the "
              f"{args.minutes:g} minute target. Nothing to do (use --fresh to regenerate).")
        return

    # One shared, numbered sentence stream. Each worker pulls the next unused sentence,
    # so concurrency never duplicates work and the seed stays tied to corpus position.
    rng = random.Random(args.seed)
    supply = enumerate(corpus(rng))
    state = {"total_seconds": total_seconds, "failures": 0, "stop": None}
    guard = threading.Lock()

    def next_sentence():
        with guard:
            if state["stop"] is not None or state["total_seconds"] >= target_seconds:
                return None
            for index, text in supply:
                if text not in already_done:
                    already_done.add(text)
                    return index, text
            return None

    def record_clip(text, seed, result):
        job_id = result["job_id"]
        duration = result["metrics"]["audio_seconds"]
        with guard:
            next_number = len(records) + 1
            while (clips_dir / f"{next_number:04d}.wav").exists():  # gap in a resumed dataset
                next_number += 1
            clip_name = f"{next_number:04d}.wav"
            # Reserve the name inside the lock; the copy below cannot be allowed to race.
            (clips_dir / clip_name).touch()
        shutil.copy(settings.data_dir / "output" / f"{job_id}.wav", clips_dir / clip_name)
        with guard:
            records.append({"file": clip_name, "text": text, "duration_seconds": duration, "seed": seed})
            state["total_seconds"] += duration
            state["failures"] = 0
            print(f"  [{len(records):4d}] {duration:5.2f}s  "
                  f"({state['total_seconds'] / 60:5.2f} / {args.minutes:g} min)  {text}")
            # Flushed every clip, so a long run that is interrupted can be resumed in place.
            write_metadata(output_dir, profile, args, records, state["total_seconds"])

    def synthesize(worker_engine):
        while True:
            item = next_sentence()
            if item is None:
                return
            index, text = item
            seed = (args.seed + index) % 2**32
            try:
                result = worker_engine.generate(profile["id"], text, speed=1.0, seed=seed)
            except T2000Error as exc:
                with guard:
                    state["failures"] += 1
                    print(f"  [skip] {text!r}: {exc}")
                    if state["failures"] >= args.max_consecutive_failures:
                        state["stop"] = (f"T-2000: {state['failures']} sentences in a row failed; "
                                         f"stopping.")
                        return
                continue
            record_clip(text, seed, result)

    print(f"T-2000: warming {args.backend} for {profile['name']!r} ({profile['id'][:12]}) "
          f"on {args.workers} worker{'s' if args.workers > 1 else ''}...")
    started = time.perf_counter()
    engines = [engine]
    for _ in range(args.workers - 1):
        engines.append(Engine(settings, ProfileStore(settings.data_dir / "profiles")))
    for worker_engine in engines:
        worker_engine.prepare_profile(profile["id"])
    print(f"T-2000: warmed in {time.perf_counter() - started:.1f}s. Generating "
          f"{(target_seconds - total_seconds) / 60:.2f} more minutes...")

    try:
        if len(engines) == 1:
            synthesize(engines[0])
        else:
            threads = [threading.Thread(target=synthesize, args=(worker_engine,), daemon=True)
                       for worker_engine in engines]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
    finally:
        write_metadata(output_dir, profile, args, records, state["total_seconds"])
        for worker_engine in engines[1:]:
            with contextlib.suppress(Exception):
                worker_engine.close()

    total_seconds = state["total_seconds"]
    if state["stop"]:
        raise SystemExit(state["stop"])

    print(f"\nT-2000: wrote {len(records)} clips, {total_seconds / 60:.2f} minutes, to {output_dir}")


if __name__ == "__main__":
    main()

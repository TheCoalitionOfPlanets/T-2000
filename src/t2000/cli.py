import argparse
from collections import Counter
import json
import logging
from pathlib import Path

from .config import Settings
from .errors import T2000Error


def print_voices(store) -> None:
    """Show what `t2000 voice` can serve, with enough detail to pick one."""
    profiles = store.list()
    if not profiles:
        print("No saved voices yet. Enroll one first:\n"
              "  t2000 enroll /path/to/reference.wav --name 'My voice'")
        return
    repeated = Counter(profile["name"].casefold() for profile in profiles)
    width = min(max(len(profile["name"]) for profile in profiles), 28)
    print(f"{'ID':<14}{'NAME':<{width + 3}}{'LENGTH':>7}  {'SAVED':<12}TRANSCRIPT")
    for profile in profiles:
        name = profile["name"]
        if len(name) > width:
            name = name[:width - 1] + "…"
        # A shared name cannot be used as a selector, so mark it here rather than
        # letting the ambiguity surface only when serving fails.
        name += " *" if repeated[profile["name"].casefold()] > 1 else ""
        transcript = profile["transcript"].replace("\n", " ").strip() or "(none)"
        if len(transcript) > 44:
            transcript = transcript[:43] + "…"
        print(f"{profile['id'][:12]:<14}{name:<{width + 3}}{profile['duration_seconds']:>6.1f}s  "
              f"{profile['created_at'][:10]:<12}{transcript}")
    print(f"\n{len(profiles)} saved voice{'s' if len(profiles) != 1 else ''}. "
          "Serve one with: t2000 voice <ID or name>")
    if any(count > 1 for count in repeated.values()):
        print("* This name is shared by more than one voice; select those by ID.")


def main():
    parser = argparse.ArgumentParser(description="T-2000 local voice studio")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Serve the local studio (use an SSH tunnel for a remote GPU)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7861)
    serve.add_argument("--preload", action="store_true", help="Load models before serving")
    voice = sub.add_parser("voice", help="Serve a standalone app for one saved voice")
    voice.add_argument("profile", nargs="?", help="Saved voice name or ID (for example: Prieyan)")
    voice.add_argument("--list", action="store_true", dest="list_voices",
                       help="List the saved voices you can serve, then exit")
    voice.add_argument("--host", default="127.0.0.1")
    voice.add_argument("--port", type=int, default=7862)
    voice.add_argument("--title", help="App title; defaults to <voice>-TTS")
    voice.add_argument("--warm", action=argparse.BooleanOptionalAction, default=True,
                       help="Load the engine and condition the voice at startup; --no-warm defers it")
    enroll = sub.add_parser("enroll", help="Save/analyze a WAV reference and optionally prepare conditioning")
    enroll.add_argument("reference", type=Path)
    enroll.add_argument("--name", default="Reference")
    enroll.add_argument("--transcript", default="")
    enroll.add_argument("--transcribe", action=argparse.BooleanOptionalAction, default=True,
                        help="Transcribe new references by default; --no-transcribe opts out")
    enroll.add_argument("--prepare", action="store_true")
    enroll.add_argument("--start-seconds", type=float, default=0, help="Keep audio from this time")
    enroll.add_argument("--end-seconds", type=float, help="Keep audio up to this time")
    enroll.add_argument("--no-noise-reduction", action="store_true", help="Keep the original background sound")
    enroll.add_argument("--auto-trim", action="store_true", help="Trim quiet edges with speech padding")
    generate = sub.add_parser("generate", help="Generate a complete WAV using an enrolled profile")
    generate.add_argument("--profile", required=True)
    generate.add_argument("--text", required=True)
    generate.add_argument("--speed", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--sentence-gap", type=float, default=0.0,
                          help="Silence inserted between sentences, in seconds.")
    sub.add_parser("doctor", help="Report runtime/model availability without loading weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.from_env()
        if args.command == "doctor":
            from .doctor import inspect_runtime

            print(json.dumps(inspect_runtime(settings), indent=2))
            return
        from .engine import Engine
        from .profiles import ProfileStore

        store = ProfileStore(settings.data_dir / "profiles")
        engine = Engine(settings, store)
        if args.command == "serve":
            import uvicorn
            from .api import create_app

            app = create_app(settings)
            if args.preload:
                app.state.engine.backend.load()
                app.state.engine.loaded = True
            uvicorn.run(app, host=args.host, port=args.port, ws_max_size=65536)
        elif args.command == "voice":
            import uvicorn
            from .solo import create_solo_app, resolve_profile

            if args.list_voices:
                print_voices(store)
                return
            if not args.profile:
                voice.error("Name a saved voice to serve, or use --list to see them.")
            selected = resolve_profile(store, args.profile)
            title = args.title.strip() if args.title and args.title.strip() else f"{selected['name']}-TTS"
            print(f"T-2000: serving {title} ({selected['name']}, {selected['id'][:12]}) "
                  f"on http://{args.host}:{args.port}")
            uvicorn.run(create_solo_app(settings, profile=selected["id"], title=title, warm=args.warm),
                        host=args.host, port=args.port, ws_max_size=65536)
        elif args.command == "enroll":
            profile = store.create(
                args.reference, args.name, args.transcript, start_seconds=args.start_seconds,
                end_seconds=args.end_seconds, noise_reduction=not args.no_noise_reduction,
                auto_trim=args.auto_trim, max_seconds=engine.max_reference_seconds,
            )
            if args.transcript.strip() and profile["transcript"] != args.transcript.strip():
                profile = store.update_transcript(profile["id"], args.transcript)
            if args.transcribe and not profile["transcript"]:
                from .stt import transcribe_file

                try:
                    profile = store.update_transcript(profile["id"], transcribe_file(
                        store.reference_path(profile["id"]), model_dir=settings.stt_model_dir,
                        num_threads=settings.num_threads, max_seconds=30,
                    ), only_if_empty=True)
                except T2000Error as exc:
                    profile = dict(profile, transcription_error=str(exc))
            if args.prepare:
                engine.prepare_profile(profile["id"])
            print(json.dumps(profile, indent=2))
        elif args.command == "generate":
            print(json.dumps(engine.generate(args.profile, args.text, speed=args.speed, seed=args.seed,
                                              sentence_gap=args.sentence_gap), indent=2))
    except (T2000Error, OSError, ValueError) as exc:
        parser.exit(1, f"T-2000: {exc}\n")


if __name__ == "__main__":
    main()

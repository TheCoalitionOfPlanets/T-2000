import argparse
import json
import logging
from pathlib import Path

from .config import Settings
from .errors import T2000Error


def main():
    parser = argparse.ArgumentParser(description="T-2000 local voice studio")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Serve the local studio (use an SSH tunnel for a remote GPU)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7861)
    serve.add_argument("--preload", action="store_true", help="Load models before serving")
    enroll = sub.add_parser("enroll", help="Save/analyze a WAV reference and optionally prepare conditioning")
    enroll.add_argument("reference", type=Path)
    enroll.add_argument("--name", default="Reference")
    enroll.add_argument("--transcript", default="")
    enroll.add_argument("--transcribe", action="store_true")
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
        elif args.command == "enroll":
            profile = store.create(args.reference, args.name, args.transcript, start_seconds=args.start_seconds,
                                   end_seconds=args.end_seconds, noise_reduction=not args.no_noise_reduction,
                                   auto_trim=args.auto_trim)
            if args.transcribe:
                from .stt import transcribe_file

                profile = store.update_transcript(profile["id"], transcribe_file(
                    store.reference_path(profile["id"]), model_dir=settings.stt_model_dir,
                    num_threads=settings.num_threads, max_seconds=30,
                ))
            if args.prepare:
                engine.prepare_profile(profile["id"])
            print(json.dumps(profile, indent=2))
        elif args.command == "generate":
            print(json.dumps(engine.generate(args.profile, args.text, speed=args.speed, seed=args.seed), indent=2))
    except (T2000Error, OSError, ValueError) as exc:
        parser.exit(1, f"T-2000: {exc}\n")


if __name__ == "__main__":
    main()

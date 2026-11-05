"""A standalone single-voice app: one saved reference, text in, speech out.

The studio enrolls and compares voices. This serves exactly one enrolled voice —
no profile list, no enrollment, no engine switching — and warms its backend and
conditioning at startup so the first request pays no load or prepare cost.
"""

import asyncio
import contextlib
import logging
import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .api import _same_origin, stream_generation
from .config import Settings
from .engine import Engine
from .errors import T2000Error
from .profiles import ProfileStore

logger = logging.getLogger(__name__)

_HEX = re.compile(r"[0-9a-f]+\Z")


class SpeechRequest(BaseModel):
    """The pinned voice is server-side; a client may only choose words and delivery."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=6000)
    speed: float = Field(default=1.0, ge=0.7, le=1.4, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=0xFFFFFFFF, strict=True)


def resolve_profile(store: ProfileStore, selector: str) -> dict:
    """Find one saved voice by full ID, unambiguous ID prefix, or saved name."""
    if not isinstance(selector, str) or not selector.strip():
        raise T2000Error("Name the saved voice to serve, by its name or ID.")
    selector = selector.strip()
    profiles = store.list()
    if not profiles:
        raise T2000Error("No voices are saved yet. Enroll a recording in the studio first.")
    exact = [profile for profile in profiles if profile["id"] == selector]
    if exact:
        return exact[0]
    lowered = selector.casefold()
    matches = [profile for profile in profiles if profile["name"].casefold() == lowered]
    if not matches and len(selector) >= 8 and _HEX.fullmatch(lowered):
        matches = [profile for profile in profiles if profile["id"].startswith(lowered)]
    if not matches:
        known = ", ".join(sorted({profile["name"] for profile in profiles})) or "none"
        raise T2000Error(f"No saved voice matches {selector!r}. Saved voices: {known}.")
    if len(matches) > 1:
        ids = ", ".join(f"{profile['name']} ({profile['id'][:12]})" for profile in matches)
        raise T2000Error(f"{selector!r} matches more than one saved voice: {ids}. Use the voice ID.")
    return matches[0]


def create_solo_app(settings: Settings | None = None, *, profile: str | None = None,
                    title: str | None = None, warm: bool = True, backend=None) -> FastAPI:
    """Serve one saved voice. `profile` is its name or ID; `title` names the app."""
    settings = settings or Settings.from_env()
    store = ProfileStore(settings.data_dir / "profiles")
    engine = Engine(settings, store, backend)
    # Resolve once at startup: the served voice cannot change while the app runs.
    voice = resolve_profile(store, profile)
    profile_id = voice["id"]
    name = title.strip() if isinstance(title, str) and title.strip() else f"{voice['name']}-TTS"
    warming = {"state": "cold", "detail": "", "seconds": None}
    warm_lock = threading.Lock()

    def run_warm() -> dict:
        """Load the backend and condition this voice; repeat calls are cheap."""
        with warm_lock:
            warming.update(state="warming", detail="")
            try:
                result = engine.prepare_profile(profile_id)
            except T2000Error as exc:
                warming.update(state="cold", detail=str(exc), seconds=None)
                raise
            except Exception:
                logger.exception("Warming %s failed", name)
                warming.update(state="cold", detail="Warming failed. Check server logs.", seconds=None)
                raise
            warming.update(state="ready", detail="",
                           seconds=result["load_seconds"] + result["prepare_seconds"])
            return result

    @contextlib.asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(asyncio.to_thread(run_warm)) if warm else None
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await asyncio.to_thread(engine.close)

    app = FastAPI(title=name, version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.store = store
    app.state.profile_id = profile_id
    # One running request plus one waiter; model calls cannot be interrupted mid-kernel.
    slots = threading.BoundedSemaphore(2)

    @app.middleware("http")
    async def origin_guard(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not _same_origin(
            request.headers.get("origin"), request.headers.get("host", "")
        ):
            return JSONResponse({"detail": "Cross-origin writes are disabled."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(T2000Error)
    async def domain_error(request: Request, exc: T2000Error):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/voice")
    def describe():
        current = store.get(profile_id)
        status = engine.status()
        return {
            "title": name, "voice": current["name"], "profile_id": profile_id,
            "transcript": current["transcript"], "duration_seconds": current["duration_seconds"],
            "created_at": current["created_at"], "backend": status["backend"],
            "loaded": status["loaded"], "busy": status["busy"],
            "speed_control": status["speed_control"], "max_text_chars": settings.max_text_chars,
            "warm": dict(warming),
        }

    @app.post("/api/warm")
    async def warm_voice():
        if not slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="The voice is busy generating speech.")
        def work():
            try:
                return run_warm()
            finally:
                slots.release()
        return await asyncio.to_thread(work)

    @app.get("/api/reference")
    def reference_audio():
        return FileResponse(store.reference_path(profile_id), media_type="audio/wav")

    @app.get("/api/output/{filename}")
    def output(filename: str):
        if not re.fullmatch(r"[a-f0-9]{32}\.wav", filename):
            raise HTTPException(status_code=404, detail="Output not found.")
        path = settings.data_dir / "output" / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Output not found.")
        return FileResponse(path, media_type="audio/wav", filename=f"{name}-{filename}")

    @app.websocket("/api/speak")
    async def speak(socket: WebSocket):
        if not _same_origin(socket.headers.get("origin"), socket.headers.get("host", "")):
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            request = SpeechRequest.model_validate(await asyncio.wait_for(socket.receive_json(), timeout=10))
        except (ValidationError, ValueError, TimeoutError, KeyError) as exc:
            await socket.send_json({"type": "error", "message": f"Invalid speech request: {exc}"})
            await socket.close(code=1008)
            return
        except WebSocketDisconnect:
            return
        await stream_generation(socket, engine, slots, {"profile_id": profile_id, **request.model_dump()})

    static = Path(__file__).parent / "static"

    @app.get("/")
    def index():
        return FileResponse(static / "solo.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=static), name="assets")
    return app

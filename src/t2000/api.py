import asyncio
import contextlib
import io
import logging
import queue
import re
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import Settings
from .audio import prepare_reference
from .engine import MAX_SENTENCE_GAP, Cancelled, Engine
from .errors import T2000Error
from .profiles import ProfileStore

logger = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    text: str = Field(min_length=1, max_length=6000)
    speed: float = Field(default=1.0, ge=0.7, le=1.4, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=0xFFFFFFFF, strict=True)
    sentence_gap: float = Field(default=0.0, ge=0, le=MAX_SENTENCE_GAP, allow_inf_nan=False)


class TranscriptUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transcript: str = Field(max_length=5000)


class BackendUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str = Field(pattern=r"^(f5-tts-v1|melo-openvoice-v2|t1000-soar|t1000-mf2)$")


def _same_origin(origin: str | None, host: str) -> bool:
    if origin is None:
        return True  # Non-browser CLI clients do not send Origin.
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and parsed.netloc == host


async def stream_generation(socket: WebSocket, engine: Engine, slots: threading.BoundedSemaphore,
                            request: dict) -> None:
    """Run one accepted, validated speech request and pump its events to the socket."""
    if not slots.acquire(blocking=False):
        await socket.send_json({"type": "error", "message": "The generation queue is full. Try again shortly."})
        await socket.close(code=1013)
        return
    events = queue.Queue(maxsize=8)
    cancel = threading.Event()
    disconnected = threading.Event()
    finished = threading.Event()

    def emit(event):
        while not disconnected.is_set():
            if event["type"] not in {"error", "cancelled", "complete"} and cancel.is_set():
                raise Cancelled("Generation cancelled.")
            try:
                events.put(event, timeout=0.1)
                return
            except queue.Full:
                continue
        raise Cancelled("Playback disconnected.")

    def work():
        try:
            result = engine.generate(**request, emit=emit, cancel=cancel)
            emit(result)
        except Cancelled:
            with contextlib.suppress(Cancelled):
                emit({"type": "cancelled", "message": "Generation cancelled."})
        except Exception as exc:
            logger.exception("Speech generation failed")
            message = str(exc) if isinstance(exc, T2000Error) else "Generation failed. Check server logs."
            with contextlib.suppress(Cancelled):
                emit({"type": "error", "message": message})
        finally:
            slots.release()
            finished.set()

    async def receive_cancel():
        try:
            while True:
                message = await socket.receive_json()
                if message == {"type": "cancel"}:
                    cancel.set()
        except (WebSocketDisconnect, ValueError, RuntimeError, KeyError):
            disconnected.set()
            cancel.set()

    worker = threading.Thread(target=work, daemon=True, name="t2000-generation")
    worker.start()
    receiver = asyncio.create_task(receive_cancel())
    try:
        while not disconnected.is_set():
            try:
                event = await asyncio.to_thread(events.get, True, 0.2)
            except queue.Empty:
                if finished.is_set():
                    break
                continue
            if cancel.is_set() and event["type"] == "audio":
                continue
            if event["type"] == "audio":
                event = dict(event)
                pcm = event.pop("pcm")
                await asyncio.wait_for(socket.send_json(event), timeout=10)
                await asyncio.wait_for(socket.send_bytes(pcm), timeout=10)
            else:
                await asyncio.wait_for(socket.send_json(event), timeout=10)
            if event["type"] in {"complete", "cancelled", "error"}:
                break
    except (WebSocketDisconnect, RuntimeError, TimeoutError):
        pass
    finally:
        cancel.set()
        disconnected.set()
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await socket.close()


def create_app(settings: Settings | None = None, backend=None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = ProfileStore(settings.data_dir / "profiles")
    engine = Engine(settings, store, backend)
    @contextlib.asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await asyncio.to_thread(engine.close)

    app = FastAPI(title="T-2000", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.store = store
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

    @app.get("/api/status")
    def status():
        return engine.status()

    @app.post("/api/backend")
    async def select_backend(update: BackendUpdate):
        acquired = 0
        try:
            for _ in range(2):
                if not slots.acquire(blocking=False):
                    raise HTTPException(status_code=409, detail="Wait for speech generation before changing engines.")
                acquired += 1
            return await asyncio.to_thread(engine.switch_backend, update.backend)
        finally:
            for _ in range(acquired):
                slots.release()

    @app.get("/api/profiles")
    def profiles():
        return store.list()

    async def receive_upload(file: UploadFile) -> Path:
        upload_dir = settings.data_dir / ".uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = None
        try:
            with tempfile.NamedTemporaryFile(dir=upload_dir, suffix=".wav", delete=False) as out:
                path = Path(out.name)
                total = 0
                while block := await file.read(64 * 1024):
                    total += len(block)
                    if total > settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Reference upload exceeds 16 MB.")
                    out.write(block)
            return path
        except BaseException:
            if path is not None:
                path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.post("/api/reference/preview")
    async def preview_reference(
        file: UploadFile = File(...), start_seconds: float = Form(0),
        end_seconds: float | None = Form(None), noise_reduction: bool = Form(True),
        auto_trim: bool = Form(False),
    ):
        """Return a disposable preview; saving uses the same deterministic processing."""
        import soundfile as sf

        path = await receive_upload(file)
        try:
            audio = await asyncio.to_thread(
                prepare_reference, path, start_seconds=start_seconds, end_seconds=end_seconds,
                noise_reduction=noise_reduction, auto_trim=auto_trim,
            )
            target = io.BytesIO()
            await asyncio.to_thread(sf.write, target, audio.samples, audio.sample_rate, format="WAV", subtype="FLOAT")
            return Response(target.getvalue(), media_type="audio/wav", headers={
                "Cache-Control": "no-store", "X-Reference-Duration": str(audio.duration_seconds),
            })
        finally:
            path.unlink(missing_ok=True)

    @app.post("/api/profiles", status_code=201)
    async def enroll(
        file: UploadFile = File(...), name: str = Form("Reference"),
        transcript: str = Form(""), transcribe: bool = Form(True),
        start_seconds: float = Form(0), end_seconds: float | None = Form(None),
        noise_reduction: bool = Form(True), auto_trim: bool = Form(False),
    ):
        if not 1 <= len(name.strip()) <= 120 or len(transcript) > 5000:
            raise T2000Error("Use a voice name of 1–120 characters and a transcript under 5000 characters.")
        path = await receive_upload(file)
        try:
            profile = await asyncio.to_thread(
                store.create, path, name.strip(), transcript.strip(), start_seconds=start_seconds,
                end_seconds=end_seconds, noise_reduction=noise_reduction, auto_trim=auto_trim,
                max_seconds=engine.max_reference_seconds,
            )
            if transcript.strip() and profile["transcript"] != transcript.strip():
                profile = await asyncio.to_thread(store.update_transcript, profile["id"], transcript.strip())
            if transcribe and not profile["transcript"]:
                from .stt import transcribe_file

                try:
                    words = await asyncio.to_thread(
                        transcribe_file, store.reference_path(profile["id"]),
                        model_dir=settings.stt_model_dir, num_threads=settings.num_threads, max_seconds=30,
                    )
                    profile = await asyncio.to_thread(store.update_transcript, profile["id"], words,
                                                      only_if_empty=True)
                except T2000Error as exc:
                    # Preserve a valid reference if optional transcription is unavailable.
                    profile = dict(profile, transcription_error=str(exc))
            return profile
        finally:
            path.unlink(missing_ok=True)

    @app.patch("/api/profiles/{profile_id}")
    def transcript_update(profile_id: str, update: TranscriptUpdate):
        return store.update_transcript(profile_id, update.transcript)

    @app.post("/api/profiles/{profile_id}/prepare")
    async def prepare(profile_id: str):
        if not slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="The generation queue is full.")
        def work():
            try:
                return engine.prepare_profile(profile_id)
            finally:
                slots.release()
        return await asyncio.to_thread(work)

    @app.get("/api/profiles/{profile_id}/audio")
    def reference_audio(profile_id: str):
        return FileResponse(store.reference_path(profile_id), media_type="audio/wav")

    @app.get("/api/output/{filename}")
    def output(filename: str):
        if not re.fullmatch(r"[a-f0-9]{32}\.wav", filename):
            raise HTTPException(status_code=404, detail="Output not found.")
        path = settings.data_dir / "output" / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Output not found.")
        return FileResponse(path, media_type="audio/wav", filename=f"t2000-{filename}")

    @app.websocket("/api/generate")
    async def generate(socket: WebSocket):
        if not _same_origin(socket.headers.get("origin"), socket.headers.get("host", "")):
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            request = GenerationRequest.model_validate(await asyncio.wait_for(socket.receive_json(), timeout=10))
        except (ValidationError, ValueError, TimeoutError, KeyError) as exc:
            await socket.send_json({"type": "error", "message": f"Invalid generation request: {exc}"})
            await socket.close(code=1008)
            return
        except WebSocketDisconnect:
            return
        await stream_generation(socket, engine, slots, request.model_dump())

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="assets")
    app.mount("/", StaticFiles(directory=static, html=True), name="studio")
    return app

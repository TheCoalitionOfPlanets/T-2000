import os
from dataclasses import dataclass
import math
from pathlib import Path

from .errors import T2000Error


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    model_dir: Path = Path("data/models")
    tts_device: str = "cpu"
    converter_device: str = "cpu"
    speaker: str = "EN-US"
    language: str = "EN"
    num_threads: int = 4
    reference_mode: str = "full"
    conversion_tau: float = 0.3
    backend: str = "melo-openvoice-v2"
    f5_python: Path = Path(".venv-f5/bin/python")
    f5_steps: int = 32
    t1000_root: Path = Path("../T-1000")
    t1000_python: Path = Path("../T-1000/.venv/bin/python")
    t1000_profile: str = "low-vram"
    t1000_runtime_source: Path | None = None
    max_upload_bytes: int = 16 * 1024 * 1024
    max_text_chars: int = 6000
    max_output_seconds: float = 600.0
    phrase_chars: int = 180

    @classmethod
    def from_env(cls):
        data = Path(os.environ.get("T2000_DATA_DIR", "data")).expanduser().resolve()
        try:
            threads = int(os.environ.get("T2000_NUM_THREADS", "4"))
        except ValueError as exc:
            raise T2000Error("T2000_NUM_THREADS must be a positive integer.") from exc
        if not 1 <= threads <= 64:
            raise T2000Error("T2000_NUM_THREADS must be between 1 and 64.")
        reference_mode = os.environ.get("T2000_REFERENCE_MODE", "full").strip().lower()
        if reference_mode not in {"full", "speech"}:
            raise T2000Error("T2000_REFERENCE_MODE must be full or speech.")
        try:
            conversion_tau = float(os.environ.get("T2000_CONVERSION_TAU", "0.3"))
        except ValueError as exc:
            raise T2000Error("T2000_CONVERSION_TAU must be a finite number between 0 and 1.") from exc
        if not math.isfinite(conversion_tau) or not 0 <= conversion_tau <= 1:
            raise T2000Error("T2000_CONVERSION_TAU must be a finite number between 0 and 1.")
        backend = os.environ.get("T2000_BACKEND", "melo-openvoice-v2").strip().lower()
        if backend not in {"melo-openvoice-v2", "f5-tts-v1", "t1000-soar", "t1000-mf2"}:
            raise T2000Error("T2000_BACKEND must be melo-openvoice-v2, f5-tts-v1, t1000-soar or t1000-mf2.")
        t1000_profile = os.environ.get("T2000_T1000_PROFILE", "low-vram")
        if t1000_profile not in {"low-vram", "high-vram"}:
            raise T2000Error("T2000_T1000_PROFILE must be low-vram or high-vram.")
        runtime_source = os.environ.get("T2000_T1000_RUNTIME_SOURCE")
        try:
            f5_steps = int(os.environ.get("T2000_F5_STEPS", "32"))
        except ValueError as exc:
            raise T2000Error("T2000_F5_STEPS must be an integer between 8 and 64.") from exc
        if not 8 <= f5_steps <= 64:
            raise T2000Error("T2000_F5_STEPS must be an integer between 8 and 64.")
        return cls(
            data_dir=data,
            model_dir=Path(os.environ.get("T2000_MODEL_DIR", str(data / "models"))).expanduser().resolve(),
            tts_device=os.environ.get("T2000_TTS_DEVICE", "cpu"),
            converter_device=os.environ.get("T2000_CONVERTER_DEVICE", "cpu"),
            speaker=os.environ.get("T2000_SPEAKER", "EN-US"),
            language=os.environ.get("T2000_LANGUAGE", "EN"),
            num_threads=threads,
            reference_mode=reference_mode, conversion_tau=conversion_tau,
            backend=backend, f5_steps=f5_steps,
            f5_python=Path(os.environ.get("T2000_F5_PYTHON", ".venv-f5/bin/python")).expanduser().absolute(),
            t1000_root=Path(os.environ.get("T2000_T1000_ROOT", "../T-1000")).expanduser().absolute(),
            t1000_python=Path(os.environ.get("T2000_T1000_PYTHON",
                                           "../T-1000/.venv/bin/python")).expanduser().absolute(),
            t1000_profile=t1000_profile,
            t1000_runtime_source=(Path(runtime_source).expanduser().absolute() if runtime_source else None),
        )

    @property
    def stt_model_dir(self):
        from .stt import MODEL_NAME

        return Path(os.environ.get("STT_MODEL_DIR", str(self.model_dir / MODEL_NAME))).expanduser()

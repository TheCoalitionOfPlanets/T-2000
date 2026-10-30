import os
from dataclasses import dataclass
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
    max_upload_bytes: int = 16 * 1024 * 1024
    max_text_chars: int = 3000
    max_output_seconds: float = 120.0
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
        return cls(
            data_dir=data,
            model_dir=Path(os.environ.get("T2000_MODEL_DIR", str(data / "models"))).expanduser().resolve(),
            tts_device=os.environ.get("T2000_TTS_DEVICE", "cpu"),
            converter_device=os.environ.get("T2000_CONVERTER_DEVICE", "cpu"),
            speaker=os.environ.get("T2000_SPEAKER", "EN-US"),
            language=os.environ.get("T2000_LANGUAGE", "EN"),
            num_threads=threads,
        )

    @property
    def stt_model_dir(self):
        from .stt import MODEL_NAME

        return Path(os.environ.get("STT_MODEL_DIR", str(self.model_dir / MODEL_NAME))).expanduser()

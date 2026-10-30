from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    base_seconds: float = 0.0
    conversion_seconds: float = 0.0


class VoiceBackend(Protocol):
    """Called by one serialized inference worker; audio is mono float32."""

    def signature(self) -> str:
        """Stable identity including model files, preprocessing and base voice."""
        ...

    def load(self) -> None: ...

    def prepare(self, reference_path: Path, cache_dir: Path) -> None:
        """Cache conditioning for an immutable normalized reference."""
        ...

    def synthesize(self, text: str, cache_dir: Path, *, speed: float, seed: int) -> AudioChunk: ...

    def info(self) -> dict: ...

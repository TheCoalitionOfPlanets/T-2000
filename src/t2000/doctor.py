import importlib.util
import platform
import shutil
import subprocess

from .config import Settings


def inspect_runtime(settings: Settings) -> dict:
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "model_dir": str(settings.model_dir), "manifest_present": (settings.model_dir / "manifest.json").is_file(),
        "tts_device": settings.tts_device, "converter_device": settings.converter_device,
        "dependencies": {name: importlib.util.find_spec(name) is not None
                         for name in ("torch", "torchaudio", "sherpa_onnx", "transformers", "soundfile")},
        "stt_models_present": all((settings.stt_model_dir / name).is_file()
                                  for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")),
        "gpu": None,
        "note": "Availability only; this does not establish model compatibility, GPU fit, or speed.",
    }
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            result["gpu"] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["gpu"] = str(exc)
    return result

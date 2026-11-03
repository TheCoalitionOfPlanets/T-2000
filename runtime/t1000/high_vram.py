"""Full-precision SOAR/MF2 loading for GPUs with room for dense BF16 weights.

The low-VRAM profile exists to fit a 4 GB card: it quantizes every language-model
projection to NF4 and shrinks the runtime's sequence budget. Both cost speed and
headroom that a 16 GB card does not need to trade away.

This profile keeps the language model dense in its compute dtype and restores the
upstream runtime limits, so one request can hold a longer reference and speak a
much longer passage without the adapter splitting it into short phrases.
"""

from pathlib import Path

import torch

from .vocoder import configure_vocoder_batching, read_vocoder_batch_size


# Upstream DotsTtsRuntime defaults. The low-VRAM profile lowers these to 128/512.
HIGH_VRAM_MAX_GENERATE_LENGTH = 500
HIGH_VRAM_MAX_SEQUENCE_LENGTH = 2048


def load_high_vram_runtime(model_path, *, precision,
                           max_generate_length=HIGH_VRAM_MAX_GENERATE_LENGTH,
                           max_sequence_length=HIGH_VRAM_MAX_SEQUENCE_LENGTH):
    """Load SOAR/MF2 with dense weights and the upstream sequence budget.

    No bitsandbytes dependency: every module keeps its normal runtime precision.
    The DiT is still fused on CPU before any CUDA transfer so the device only
    receives final weights, exactly as the low-VRAM profile does.
    """
    compute_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(precision)
    if compute_dtype is None:
        raise ValueError("The high-VRAM profile requires bfloat16 or float16 precision.")
    for name, value in (("max_generate_length", max_generate_length),
                        ("max_sequence_length", max_sequence_length)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if max_generate_length > max_sequence_length:
        raise ValueError("max_generate_length cannot exceed max_sequence_length.")
    vocoder_batch_size = read_vocoder_batch_size()

    from dots_tts.models.dots_tts.model import DotsTtsModel
    from dots_tts.modules.backbone.dit_inference import fuse_dit_for_inference
    from dots_tts.runtime import DotsTtsRuntime

    model_path = Path(model_path)
    # Lazy inference adapters contain mutable tensors: create ordinary tensors
    # even when the caller enters inference_mode before loading the runtime.
    with torch.inference_mode(False):
        with torch.device("cpu"):
            model = DotsTtsModel.from_pretrained(model_path)
            # Upstream otherwise fuses on the first decode, temporarily holding
            # original adaLN weights, a fused Linear and its concatenated copy
            # on CUDA. Fusing on CPU keeps that transient off the GPU.
            fuse_dit_for_inference(model.core)

        runtime = DotsTtsRuntime(
            model,
            model_path,
            precision=precision,
            optimize=False,
            max_generate_length=max_generate_length,
            max_sequence_length=max_sequence_length,
            vocoder_merge_steps=1,
            warmup_on_optimize=False,
        )
        return configure_vocoder_batching(runtime, vocoder_batch_size)

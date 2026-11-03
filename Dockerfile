# syntax=docker/dockerfile:1
#
# Self-contained T-1000 (SOAR + MF2) voice studio for a 16 GB CUDA GPU.
#
# Design note - why this starts from an official PyTorch image rather than
# installing Torch with pip:
#
# On the target host, bytes fetched by the Docker daemon and bytes fetched from
# inside a build container do not travel the same path, and they do not perform
# the same. Measured on that machine, downloading the identical 848 MB Torch
# wheel:
#
#     Windows host                    ~105 MB/s
#     Inside the WSL2 VM (daemon)     848 MB in 8.2 s
#     Inside a build container        collapsed to ~0.3 MB/s after a short burst
#
# A pip install of torch+cu128 pulls roughly 6 GB of CUDA wheels through the
# slow path and stalls for over half an hour. Starting from an image that
# already contains them moves those 6 GB onto the daemon's pull, which is
# resumable, content-addressed, parallel, and fast. Only ~400 MB of small
# wheels are then fetched inside the container, which stays inside the burst
# window.
#
# The base supplies Python 3.11, torch 2.8.0+cu128, torchaudio 2.8.0+cu128,
# CUDA 12.8 and cuDNN 9. dots.tts requires torch>=2.8.0, so pip treats it as
# satisfied and never re-downloads it.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS runtime

# ffmpeg/libsndfile for audio I/O, git for the pinned dots.tts source install,
# g++ for any dependency without a prebuilt wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates ffmpeg libsndfile1 libgomp1 git g++ \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 studio \
    && useradd --uid 1000 --gid 1000 --create-home studio

# A short socket timeout with many retries suits a link that stalls rather than
# slows: a dead connection is dropped and retried instead of hanging for
# minutes. --timeout is idle time, so it never aborts a transfer still moving.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=10

WORKDIR /app

# --- Synthesis environment -------------------------------------------------
# Installed into the base image's own interpreter, which already owns Torch and
# the CUDA runtime. This is the environment the T-1000 worker subprocess runs.
COPY requirements/t1000-worker-cuda.txt /tmp/t1000-worker-cuda.txt
RUN --mount=type=cache,id=t2000-pip,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR /opt/conda/bin/python -m pip install \
        -r /tmp/t1000-worker-cuda.txt \
    && /opt/conda/bin/python -c "import torch, torchaudio, importlib.metadata as m; \
print('torch', torch.__version__, 'torchaudio', torchaudio.__version__, \
'dots-tts', m.version('dots-tts'), 'hf_hub', m.version('huggingface_hub')); \
assert torch.__version__.endswith('+cu128'), torch.__version__"
# Importing dots_tts is deliberately avoided here: the build has no GPU, and
# the metadata check already proves the install resolved.

# --- Studio environment ----------------------------------------------------
# A separate venv on the same Python 3.11. The studio pins numpy<2 while the
# synthesis stack needs numpy 2.x, so the two cannot share site-packages. The
# studio imports no Torch at all; synthesis is always a subprocess.
COPY requirements/constraints.txt /tmp/constraints.txt
RUN /opt/conda/bin/python -m venv /opt/app \
    && /opt/app/bin/python -m pip install --upgrade pip setuptools wheel

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN --mount=type=cache,id=t2000-pip,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR /opt/app/bin/python -m pip install \
        -c /tmp/constraints.txt -e '.[stt]'

# --- Application files -----------------------------------------------------
# Late, so editing them does not invalidate any dependency layer.
COPY scripts/ ./scripts/
COPY runtime/ ./runtime/
COPY requirements/ ./requirements/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/data /app/.cache \
    && chown -R 1000:1000 /app

ENV HOME=/tmp \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    NUMBA_CACHE_DIR=/app/.cache/numba \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    T2000_DATA_DIR=/app/data \
    T2000_MODEL_DIR=/app/data/models \
    T2000_APP_PYTHON=/opt/app/bin/python \
    T2000_T1000_PYTHON=/opt/conda/bin/python \
    T2000_T1000_ROOT=/app \
    T2000_T1000_RUNTIME_SOURCE=/app/runtime \
    T2000_T1000_PROFILE=high-vram \
    T2000_BACKEND=t1000-mf2 \
    T2000_TTS_DEVICE=cuda:0 \
    T2000_CONVERTER_DEVICE=cuda:0 \
    STT_MODEL_DIR=/app/data/models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER 1000:1000
EXPOSE 7861
# The first start downloads about 10 GB of weights, so the container is
# legitimately not serving for a while. Unhealthy does not trigger a restart.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60m --retries=3 \
    CMD ["/opt/app/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7861/api/status', timeout=5).read()"]
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve", "--host", "0.0.0.0", "--port", "7861"]

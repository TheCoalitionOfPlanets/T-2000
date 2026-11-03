# syntax=docker/dockerfile:1
#
# Self-contained T-1000 (SOAR + MF2) studio for a 16 GB CUDA GPU.
#
# One CUDA stack only. The studio process needs no Torch at all: synthesis runs
# in the isolated /opt/t1000 environment, which owns the single CUDA 12.8 build.
# Model weights are not baked in; the entrypoint fetches and verifies them into
# the data volume on first start.

FROM python:3.11-slim-bookworm AS base
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 -o Acquire::https::Timeout=120 update \
    && apt-get -o Acquire::Retries=5 -o Acquire::https::Timeout=120 \
       install -y --no-install-recommends ca-certificates ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 studio \
    && useradd --uid 1000 --gid 1000 --create-home studio
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

# The studio itself: FastAPI, audio I/O and CPU speech recognition. No Torch.
FROM base AS app-build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 UV_LINK_MODE=copy
WORKDIR /app
COPY requirements/ ./requirements/
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN --mount=type=cache,id=t2000-uv,target=/root/.cache/uv \
    uv venv --python /usr/local/bin/python /opt/app \
    && uv pip install --python /opt/app/bin/python \
       -c requirements/constraints.txt -e '.[stt]' \
    && uv pip install --python /opt/app/bin/python 'huggingface-hub>=0.26,<1'

# Synthesis environment: CUDA 12.8 Torch plus the pinned dots.tts runtime.
# bitsandbytes is omitted; the high-VRAM profile keeps weights dense.
FROM base AS t1000-build
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends git g++ \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 \
    UV_LINK_MODE=copy UV_CONCURRENT_DOWNLOADS=4 \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5
COPY requirements/t1000-worker-cuda.txt /tmp/t1000-worker.txt
# uv creates the environment, but pip performs the downloads. uv's HTTP/2
# client stalls indefinitely against download.pytorch.org behind some
# inspecting corporate proxies: the process sleeps with its connections open
# and makes no progress, while HTTP/1.1 to the same host from the same machine
# runs at full speed. pip speaks HTTP/1.1 and is unaffected.
RUN --mount=type=cache,id=t2000-pip,target=/root/.cache/pip \
    uv venv --seed --python /usr/local/bin/python /opt/t1000
RUN --mount=type=cache,id=t2000-pip,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR /opt/t1000/bin/python -m pip install \
       --index-url https://download.pytorch.org/whl/cu128 \
       'torch==2.8.0+cu128' 'torchaudio==2.8.0+cu128'
RUN --mount=type=cache,id=t2000-pip,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR /opt/t1000/bin/python -m pip install -r /tmp/t1000-worker.txt

FROM base AS runtime
WORKDIR /app
COPY --from=app-build /opt/app/ /opt/app/
COPY --from=t1000-build /opt/t1000/ /opt/t1000/
COPY --chown=1000:1000 pyproject.toml README.md ./
COPY --chown=1000:1000 src/ ./src/
COPY --chown=1000:1000 scripts/ ./scripts/
COPY --chown=1000:1000 runtime/ ./runtime/
COPY --chown=1000:1000 requirements/ ./requirements/
COPY --chown=1000:1000 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/data /app/.cache \
    && chown -R 1000:1000 /app/data /app/.cache
ENV PATH="/opt/app/bin:$PATH" \
    HOME=/tmp \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    NUMBA_CACHE_DIR=/app/.cache/numba \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    T2000_DATA_DIR=/app/data \
    T2000_MODEL_DIR=/app/data/models \
    T2000_APP_PYTHON=/opt/app/bin/python \
    T2000_T1000_PYTHON=/opt/t1000/bin/python \
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
# The first start downloads about 10 GB, so allow a long start period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45m --retries=3 \
    CMD ["/opt/app/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7861/api/status', timeout=5).read()"]
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve", "--host", "0.0.0.0", "--port", "7861"]

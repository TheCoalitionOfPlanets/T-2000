# syntax=docker/dockerfile:1

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

# Share the validated CUDA 12.1 wheels between Melo and F5. Their remaining
# dependencies live in separate environments, including their different SciPy.
FROM base AS shared-build
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=5
# The official index serves CUDA dependencies from NVIDIA's wheel host, too.
# Build-only caches retain completed downloads across interrupted installs.
RUN --mount=type=cache,id=t2000-pip,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    'torch==2.5.1+cu121' 'torchaudio==2.5.1+cu121'

FROM shared-build AS synthesis-build
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements/ ./requirements/
RUN --mount=type=cache,id=t2000-pip,target=/root/.cache/pip \
    python -m venv --system-site-packages /opt/melo \
    && sed '/^torch @ /d; /^torchaudio @ /d' requirements/inference-gpu.txt > requirements/melo-container.txt \
    && env -u PIP_NO_CACHE_DIR /opt/melo/bin/python -m pip install -r requirements/melo-container.txt \
    && python -m venv --system-site-packages /opt/f5 \
    && sed '/^torch @ /d; /^torchaudio @ /d' requirements/inference-f5.txt > requirements/f5-container.txt \
    && env -u PIP_NO_CACHE_DIR /opt/f5/bin/python -m pip install -r requirements/f5-container.txt
COPY pyproject.toml README.md ./
COPY src/ ./src/
# Keep the editable source location: backend adapters locate /app/scripts by
# walking up from their source files.
RUN --mount=type=cache,id=t2000-pip,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR /opt/melo/bin/python -m pip install -c requirements/constraints.txt -e '.[stt]'

FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

# T-1000 needs CUDA 12.8 / Torch 2.8, NumPy 2 and a newer Transformers. Its
# environment must not inherit the shared Melo/F5 site-packages.
FROM base AS t1000-build
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends git g++ \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_DOWNLOADS=never UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 \
    UV_LINK_MODE=copy UV_CONCURRENT_DOWNLOADS=4
COPY requirements/t1000-worker.txt /tmp/t1000-worker.txt
RUN --mount=type=cache,id=t2000-uv,target=/root/.cache/uv \
    uv venv --python /usr/local/bin/python /opt/t1000 \
    && uv pip install --python /opt/t1000/bin/python \
       --index-url https://download.pytorch.org/whl/cu128 \
       'torch==2.8.0+cu128' 'torchaudio==2.8.0+cu128'
RUN --mount=type=cache,id=t2000-uv,target=/root/.cache/uv \
    uv pip install --python /opt/t1000/bin/python \
       -r /tmp/t1000-worker.txt

FROM base AS runtime-lite
WORKDIR /app
COPY --from=shared-build /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=synthesis-build /opt/melo/ /opt/melo/
COPY --from=synthesis-build /opt/f5/ /opt/f5/
COPY --chown=1000:1000 pyproject.toml README.md ./
COPY --chown=1000:1000 src/ ./src/
COPY --chown=1000:1000 scripts/ ./scripts/
COPY --chown=1000:1000 requirements/ ./requirements/
RUN mkdir -p /app/data /app/.cache \
    && chown -R 1000:1000 /app/data /app/.cache
ENV PATH="/opt/melo/bin:$PATH" \
    HOME=/tmp \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    NUMBA_CACHE_DIR=/app/.cache/numba \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    T2000_DATA_DIR=/app/data \
    T2000_MODEL_DIR=/app/data/models \
    T2000_F5_PYTHON=/opt/f5/bin/python \
    T2000_T1000_ROOT=/t1000 \
    T2000_T1000_PYTHON=/opt/t1000/bin/python \
    T2000_TTS_DEVICE=cuda:0 \
    T2000_CONVERTER_DEVICE=cuda:0 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER 1000:1000
EXPOSE 7861
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["/opt/melo/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7861/api/status', timeout=5).read()"]
ENTRYPOINT ["/opt/melo/bin/t2000"]
CMD ["serve", "--host", "0.0.0.0", "--port", "7861"]

FROM runtime-lite AS runtime
COPY --from=t1000-build /opt/t1000/ /opt/t1000/

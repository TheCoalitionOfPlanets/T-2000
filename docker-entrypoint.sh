#!/usr/bin/env bash
# Provision the pinned models on first start, then run the studio.
#
# Everything lands under T2000_DATA_DIR, which Compose keeps in a named volume,
# so the cost is paid once and survives image rebuilds. Complete model
# directories are verified and skipped. Set T2000_SKIP_MODEL_FETCH=1 to start
# strictly offline against whatever is already present.
set -euo pipefail

data_dir="${T2000_DATA_DIR:-/app/data}"
models_dir="${T2000_MODEL_DIR:-$data_dir/models}"
worker_python="${T2000_T1000_PYTHON:-/opt/conda/bin/python}"
app_python="${T2000_APP_PYTHON:-/opt/app/bin/python}"
stt_dir="${STT_MODEL_DIR:-$models_dir/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25}"

log() { printf '[t2000-setup] %s\n' "$*"; }

if [[ "${T2000_SKIP_MODEL_FETCH:-0}" != "1" ]]; then
    mkdir -p "$models_dir"

    # --check needs no network and no huggingface_hub; it only stats the files
    # and hashes the checkpoint against the pin the backend enforces at load.
    if "$app_python" /app/scripts/setup_t1000.py --models-dir "$models_dir" --check >/dev/null 2>&1; then
        log "dots.tts checkpoints present and verified"
    else
        log "fetching dots.tts SOAR + MF2 checkpoints (about 9.3 GB, once)"
        log "this is the long step; later starts reuse the volume"
        # The image is offline by default, so the fetchers need that lifted.
        # hf_transfer splits each file across many ranged connections, which
        # matters on links that throttle a single long-lived stream.
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HUB_ENABLE_HF_TRANSFER=1 \
            "$worker_python" /app/scripts/setup_t1000.py --models-dir "$models_dir"
    fi

    if [[ -f "$stt_dir/encoder.int8.onnx" ]]; then
        log "speech-recognition model present"
    else
        log "fetching the speech-recognition model for reference transcripts (about 630 MB, once)"
        "$app_python" /app/scripts/fetch_stt.py --model-dir "$stt_dir" \
            --cache-dir "${XDG_CACHE_HOME:-/app/.cache}/stt"
    fi

    log "models ready"
fi

exec /opt/app/bin/t2000 "$@"

#!/usr/bin/env bash
# Provision the pinned models on first start, then run the studio.
#
# Everything lands under T2000_DATA_DIR, which Compose keeps in a named volume,
# so this cost is paid once and survives image rebuilds. Already-complete model
# directories are verified and skipped. Set T2000_SKIP_MODEL_FETCH=1 to run
# strictly offline against whatever is already present.
set -euo pipefail

data_dir="${T2000_DATA_DIR:-/app/data}"
models_dir="${T2000_MODEL_DIR:-$data_dir/models}"
t1000_python="${T2000_T1000_PYTHON:-/opt/t1000/bin/python}"
app_python="${T2000_APP_PYTHON:-/opt/app/bin/python}"

log() { printf '[t2000-setup] %s\n' "$*"; }

if [[ "${T2000_SKIP_MODEL_FETCH:-0}" != "1" ]]; then
    mkdir -p "$models_dir"
    # The runtime image is offline by default; the fetchers need that lifted.
    if ! "$app_python" /app/scripts/setup_t1000.py --models-dir "$models_dir" --check >/dev/null 2>&1; then
        log "fetching dots.tts SOAR + MF2 checkpoints (about 9.3 GB, once)"
        env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
            "$t1000_python" /app/scripts/setup_t1000.py --models-dir "$models_dir"
    else
        log "dots.tts checkpoints present and verified"
    fi

    stt_dir="${STT_MODEL_DIR:-$models_dir/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25}"
    if [[ ! -f "$stt_dir/encoder.int8.onnx" ]]; then
        log "fetching the speech-recognition model for reference transcripts (about 630 MB, once)"
        "$app_python" /app/scripts/fetch_stt.py --model-dir "$stt_dir" \
            --cache-dir "${XDG_CACHE_HOME:-/app/.cache}/stt"
    else
        log "speech-recognition model present"
    fi
    log "models ready"
fi

exec /opt/app/bin/t2000 "$@"

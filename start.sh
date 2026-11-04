#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$project_root"

force_build=false
high_vram=false
usage() {
    printf 'Usage: %s [--build] [--high-vram]\n' "$0"
    printf 'Start the shared GPU studio, reusing its image or recovery archive.\n'
    printf 'Use --high-vram for the separate self-contained 16 GB GPU deployment.\n'
    printf 'Use --build to refresh the selected image before starting.\n'
}
while (( $# )); do
    case "$1" in
        --build) force_build=true ;;
        --high-vram) high_vram=true ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done

compose=(docker compose)
image=t2000:gpu
required_gib=60
wait_timeout=180
if "$high_vram"; then
    compose+=(-p t2000-high-vram -f compose.high-vram.yaml)
    image=t2000:high-vram
    required_gib=40
    wait_timeout=3600
fi

"${compose[@]}" config --quiet
build_image() {
    local available_kib
    available_kib=$(df -Pk "$project_root" | awk 'NR == 2 { print $4 }')
    if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || (( available_kib < required_gib * 1024 * 1024 )); then
        printf 'A fresh image build requires %s GiB free on this host.\n' "$required_gib" >&2
        printf 'Free build space first; normal startup can reuse an installed image.\n' >&2
        exit 1
    fi
    "${compose[@]}" build studio
}

# Finish the build before releasing a running shared studio's GPU.
recovery_archive="$project_root/.cache/t2000-gpu.tar"
if "$force_build"; then
    build_image
elif ! docker image inspect "$image" >/dev/null 2>&1; then
    if ! "$high_vram" && [[ -f "$recovery_archive" ]]; then
        docker image load -i "$recovery_archive"
        if ! docker image inspect "$image" >/dev/null 2>&1; then
            printf 'The recovery archive did not contain the expected %s image.\n' "$image" >&2
            exit 1
        fi
    else
        build_image
    fi
fi

if ! "$high_vram" && command -v systemctl >/dev/null 2>&1; then
    for unit in t2000.service t1000-3050.service; do
        if systemctl --user cat "$unit" >/dev/null 2>&1; then
            systemctl --user stop "$unit"
        fi
    done
fi

"${compose[@]}" up -d --no-build --pull never --wait --wait-timeout "$wait_timeout" studio
address=$("${compose[@]}" port studio 7861)
printf 'T-2000 is ready at http://%s\n' "$address"

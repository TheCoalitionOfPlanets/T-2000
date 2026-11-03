#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$project_root"

force_build=false
case "${1:-}" in
    "") ;;
    --build) force_build=true; shift ;;
    --help|-h)
        printf 'Usage: %s [--build]\n' "$0"
        printf 'Start T-2000 using its installed image or local recovery archive.\n'
        printf 'Use --build to refresh the image from the canonical Dockerfile.\n'
        exit 0
        ;;
    *) printf 'Usage: %s [--build]\n' "$0" >&2; exit 2 ;;
esac
if (( $# )); then
    printf 'Usage: %s [--build]\n' "$0" >&2
    exit 2
fi

docker compose config --quiet
build_image() {
    local available_kib
    available_kib=$(df -Pk "$project_root" | awk 'NR == 2 { print $4 }')
    if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || (( available_kib < 40 * 1024 * 1024 )); then
        printf 'A fresh image build plus the model download requires 40 GiB free on this host.\n' >&2
        printf 'Free build space first; normal startup can reuse the image or recovery archive.\n' >&2
        exit 1
    fi
    docker compose build studio
}
# Finish the build before releasing a running studio's GPU.
recovery_archive="$project_root/.cache/t2000-gpu.tar"
if "$force_build"; then
    build_image
elif ! docker image inspect t2000:gpu >/dev/null 2>&1; then
    if [[ -f "$recovery_archive" ]]; then
        docker image load -i "$recovery_archive"
        if ! docker image inspect t2000:gpu >/dev/null 2>&1; then
            printf 'The recovery archive did not contain the expected t2000:gpu image.\n' >&2
            exit 1
        fi
    else
        build_image
    fi
fi

if command -v systemctl >/dev/null 2>&1; then
    for unit in t2000.service t1000-3050.service; do
        if systemctl --user cat "$unit" >/dev/null 2>&1; then
            systemctl --user stop "$unit"
        fi
    done
fi

docker compose up -d --no-build --pull never --wait --wait-timeout 180 studio
address=$(docker compose port studio 7861)
printf 'T-2000 is ready at http://%s\n' "$address"

# Self-contained T-1000 studio for a 16 GB GPU

Use this deployment on a fresh Linux x86_64 machine, or Windows with Docker
Desktop/WSL2, with an NVIDIA GPU, Docker Compose and NVIDIA container support.
The image supplies T-1000 SOAR and MF2, their runtime sources and dependencies.
The first start downloads pinned synthesis and speech-recognition models.

The prepared 4 GB host continues to use the default [four-engine deployment](docker.md).
The two deployments have separate recipes, image tags, Compose project names
and volumes:

| | Prepared 4 GB host | Self-contained 16 GB host |
|---|---|---|
| Build recipe | `Dockerfile` | `Dockerfile.high-vram` |
| Compose file | `compose.yaml` | `compose.high-vram.yaml` |
| Image | `t2000:gpu` | `t2000:high-vram` |
| Compose project | `t2000` | `t2000-high-vram` |
| Engines | Melo/OpenVoice, F5, T-1000 SOAR/MF2 | T-1000 SOAR/MF2 |
| Data | Prepared host paths | Named volumes |
| Launcher | `./start.sh` | `./start.sh --high-vram` or `start.bat` |

Both publish port 7861 on loopback by default. Run only one deployment on that
port at a time, or override `T2000_PORT`. The high-VRAM launcher does not stop
native studios; release any other synthesis model's GPU allocation first.

## Start

```sh
./start.sh --high-vram
```

On Windows, run `start.bat`. Both launchers build the image when absent and
allow up to an hour for provisioning and HTTP health. Add `--build` to refresh
an installed image: `./start.sh --high-vram --build` or `start.bat --build`.
The high-VRAM launcher never loads the 4 GB host's recovery archive.

The equivalent direct Compose command starts in the background:

```sh
docker compose -f compose.high-vram.yaml up -d --build
docker compose -f compose.high-vram.yaml logs -f studio
```

First start fetches roughly 10 GB before the server answers. Without `--wait`,
the command returns while those downloads continue. To wait explicitly, use
`--wait --wait-timeout 3600`. Open <http://127.0.0.1:7861> once the container is
healthy. A healthy HTTP endpoint does not establish that synthesis fits the GPU;
prepare a voice and generate speech to validate it on the intended hardware.

## Models and persistent data

The entrypoint writes these pinned resources into the Compose project's
`models` volume, mounted at `/app/data`:

| Resource | Source |
|---|---|
| SOAR synthesis | `dots-studio/dots.tts-soar` at the adapter's pinned revision |
| MF2 synthesis | `dots-studio/dots.tts-mf-2steps` at the adapter's pinned revision |
| CPU speech recognition | Pinned Nemotron INT8 `sherpa-onnx` release archive |

Synthesis downloads go directly into `/app/data/models` without a second hub
cache copy. `scripts/setup_t1000.py` verifies required files and the checkpoint
SHA-256 against the adapter's pins. Speech recognition uses the existing
archive checksum and installer. Completed models are reused on restart.
`T2000_SKIP_MODEL_FETCH=1` skips provisioning for an offline start with models
already present.

Saved voices, recordings and generated WAVs also live under `/app/data`.
The separate `cache` volume holds Hugging Face, numba and compiler caches.
Stopping or rebuilding the container preserves both volumes. Adding `-v` to
`docker compose -f compose.high-vram.yaml down` deletes them, including saved audio.

## Runtime and profile

`Dockerfile.high-vram` starts from
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`. Torch, TorchAudio and CUDA arrive
as image layers instead of a separate wheel installation during the build.

| Environment | Contents |
|---|---|
| `/opt/app` | FastAPI studio, audio I/O, NumPy 1.x and CPU speech recognition |
| `/opt/conda` | Torch 2.8.0+cu128, NumPy 2.x and the pinned `dots.tts` worker |
| `/app/runtime/t1000` | Vendored loaders, prompt handling and vocoder batching |

The worker uses dense BF16 language-model weights, a 2048-token sequence
budget and up to 500 generated patches. Complete reference recordings may be
1–10 seconds long. The 4 GB deployment keeps its quantized low-VRAM profile,
512/128 sequence/generation budgets and six-second reference limit.
The high-VRAM image does not install bitsandbytes and fixes its Compose
profile to `high-vram`.

## Configuration and operations

Every setting has a default. On the high-VRAM host, optionally copy
`.env.high-vram.example` to `.env` to change the port, SOAR/MF2 selection,
device, thread count, network MTU or offline provisioning setting.
Allow **40 GiB free** for image layers, build caches and models; the Linux
launcher checks available project-filesystem space before a build. The Windows
launcher leaves disk-space management to Docker Desktop.

The container runs as UID/GID 1000 with a read-only application filesystem,
all capabilities dropped, a 2 GB `/tmp` tmpfs, a 1024-process limit and 1 GB
of shared memory. Logs rotate at 30 MB. The bridge MTU defaults to 1280 and
can be overridden with `T2000_MTU`.

```sh
docker compose -f compose.high-vram.yaml ps
docker compose -f compose.high-vram.yaml logs --tail=100 studio
docker compose -f compose.high-vram.yaml exec studio /opt/app/bin/t2000 doctor
docker compose -f compose.high-vram.yaml stop studio
docker compose -f compose.high-vram.yaml down
```

Fresh image builds and GPU inference for this profile remain unverified in the
merge environment. Python regression tests and Compose validation cover paths,
profile wiring and startup behavior; a real first-volume download, voice
preparation and synthesis smoke test are still needed on the 16 GB host.

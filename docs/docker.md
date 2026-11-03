# Docker: self-contained T-1000 studio

One command on a machine with an NVIDIA GPU, Docker Engine with the Compose
plugin, and the NVIDIA Container Toolkit:

```sh
docker compose up -d --build
```

Nothing else is prepared first. No host checkout of T-1000, no virtual
environment, no manual model download, and no `.env`. The studio answers on
<http://127.0.0.1:7861>.

## What the image contains

The image ships **T-1000 SOAR and MF2 only**, and carries a single CUDA stack:

| Environment | Contents |
|---|---|
| `/opt/app` | The studio: FastAPI, audio I/O, CPU speech recognition. **No Torch.** |
| `/opt/t1000` | Synthesis: Torch 2.8.0+cu128 and the pinned `dots.tts` runtime |
| `/app/runtime/t1000` | The vendored low/high-VRAM loaders, prompt cache and vocoder batching |

The studio process itself never imports Torch; synthesis runs in the isolated
`/opt/t1000` environment behind a JSON-lines worker. That is why one CUDA build
is enough.

The `dots.tts` package installs from its pinned upstream commit, and the
vendored loaders live in `runtime/t1000/`. The adapter hashes those four files
into the backend signature, so a change to them invalidates a loaded model
rather than silently altering synthesis.

## First start

Weights are not baked into the image. The entrypoint provisions them into the
`models` volume on first start and verifies each one:

| Resource | Size | Source |
|---|---|---|
| `dots.tts-soar` | 5.2 GB | `dots-studio/dots.tts-soar`, pinned revision, Apache-2.0 |
| `dots.tts-mf-2steps` | 4.1 GB | `dots-studio/dots.tts-mf-2steps`, pinned revision, Apache-2.0 |
| Nemotron INT8 speech recognition | 630 MB | pinned `sherpa-onnx` release asset |

The checkpoint SHA-256 is checked against the same pins the backend enforces at
load time; a mismatch aborts rather than running an unverified model. Expect
roughly 10 GB of download once. Follow it with:

```sh
docker compose logs -f studio
```

Completed directories are detected and skipped, so restarts and image rebuilds
never re-download. The health check allows a 45-minute start period to cover the
first fetch. To start strictly offline against what is already present, set
`T2000_SKIP_MODEL_FETCH=1`.

## VRAM profiles

`T2000_T1000_PROFILE` selects how the language model is loaded. The default is
`high-vram`.

| | `high-vram` (default) | `low-vram` |
|---|---|---|
| Language-model weights | dense BF16 | NF4 via bitsandbytes |
| `max_sequence_length` | 2048 (upstream) | 512 |
| `max_generate_length` | 500 patches (~80 s) | 128 patches (~20.5 s) |
| Reference limit | 10 s | 6 s |
| Needs bitsandbytes | no | yes |

`high-vram` removes the per-matmul NF4 dequantization from autoregressive
decoding, which is the dominant cost in the quantized profile. Restoring the
upstream sequence budget lets one request hold a longer reference and speak a
longer passage, so the delivery planner splits text into fewer phrases and the
reference is re-prefilled fewer times.

The reference limit rises to 10 s because that is the model's own ceiling:
`config.json` sets `xvec_max_audio_seconds: 10.0`, and beyond it the speaker
encoder takes a **random** crop and stops caching the embedding. A longer clip
would make identity vary with the seed instead of improving it.

The default image does not install bitsandbytes, so `low-vram` needs
`requirements/t1000-worker.txt` in the worker environment instead of
`requirements/t1000-worker-cuda.txt`.

## Disk and build

The build needs Internet access and temporary space for one CUDA wheel set.
Allow **40 GiB free** for the image, BuildKit caches and the model volume;
`start.sh` checks this before a full build. CUDA wheels come from the official
[PyTorch index](https://pytorch.org/get-started/previous-versions/) with
five-minute timeouts, and BuildKit caches keep completed downloads so an
interrupted build resumes rather than restarting.

### If the build appears to hang on the Torch step

uv's HTTP/2 client can stall indefinitely against `download.pytorch.org` through
an inspecting corporate proxy: the process sleeps with its connections open and
makes no progress, while HTTP/1.1 to the same host from the same machine runs at
full speed. The CUDA install therefore uses pip, which speaks HTTP/1.1. If a
step ever looks stuck, confirm it with CPU time rather than wall time:

```sh
docker buildx history ls                     # is the build still Running?
docker system df                             # has Build Cache grown?
```

A step consuming almost no CPU over many minutes is blocked, not working.

A rebuild is only needed when dependencies change. The studio source is
installed editable against `/app/src`, so mounting the working tree makes code
changes live on restart:

```yaml
    volumes:
      - type: bind
        source: ./src
        target: /app/src
        read_only: true
```

## Data and hardening

`models:/app/data` holds weights, saved voices, recordings and generated WAVs;
`cache:/app/.cache` holds the Hugging Face and numba caches. Both are named
volumes, so a fresh clone needs no host directories prepared. The container runs
read-only as UID 1000 with all capabilities dropped, `no-new-privileges`, a
512-process limit, a 2 GB tmpfs on `/tmp`, and the port published only on
loopback.

## Overrides

Copy `.env.docker.example` to `.env` to change the port, engine, profile, device
or thread count. Every value has a working default.

```sh
docker compose up -d                 # start, reusing the built image
docker compose logs -f studio        # watch the first model fetch
docker compose restart studio        # pick up source changes when src is mounted
docker compose down                  # stop; volumes and models persist
```

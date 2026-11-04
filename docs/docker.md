# Docker on the GPU host

This is the prepared 4 GB deployment and remains the default. For a fresh
16 GB GPU machine, see the separate [self-contained deployment](docker-high-vram.md).

The Docker studio includes isolated Melo/OpenVoice, F5, and T-1000 workers. Melo
and F5 share the pinned Torch 2.5.1 CUDA 12.1 libraries; T-1000 has its own Torch
2.8.0 CUDA 12.8 environment. The build includes code and dependencies, while
models, recorded references, and generated audio stay on the host.

Use Linux x86_64, Docker Engine with the Compose plugin, and the NVIDIA Container
Toolkit configured for Docker. GPU access uses Docker's
[Compose device reservation](https://docs.docker.com/compose/how-tos/gpu-support/).
The first full build needs Internet access and substantial temporary disk space;
allow **60 GiB free** beyond the existing models for the full image,
dependency caches, temporary layers, and image export/loading. Later starts
reuse the installed image.
The launcher checks this free-space requirement before a full build and leaves
the running studios alone if there is not enough space. It does not apply to
normal startup using the installed image or the recovery archive.
The build selects CUDA wheels and their dependencies through the official
[PyTorch indexes](https://pytorch.org/get-started/previous-versions/), uses
five-minute HTTP timeouts, and keeps completed downloads in BuildKit caches so
an interrupted install can reuse them. These download caches are outside the
runtime image and can be reclaimed after the image is verified.

The configured SSH server keeps T-2000 at `/home/prieyan/weeb/T-2000`.
From the T-2000 directory, copy `.env.docker.example` to `.env`. Set
`T2000_T1000_HOST_ROOT` to the existing T-1000 checkout on the Docker host, for
example `/home/prieyan/weeb/T-1000`. Its `apps/api/src/t1000` source and verified
`data/models/dots.tts-soar` and `data/models/dots.tts-mf-2steps` models are mounted
read-only. Only those source and model directories enter the container; the
T-1000 Git repository and native environment stay outside it. The image runs as
UID/GID **1000:1000**, matching the configured server and its `data` directory.
Moving to a host with different ownership requires adapting the image's user
and file ownership before building it.

Keep the prepared T-2000 models in `data/models`, including the synthesis
manifests, source bundles, and Nemotron recognition model. Docker startup does
not redownload or replace models. On a fresh machine, first prepare or transfer
them as described in [model setup](models.md) and the README's F5 instructions.

Start the studio with one command:

```sh
./start.sh
```

This validates the configuration, uses the installed image, restores it from
`.cache/t2000-gpu.tar` if missing, or builds it if neither is available. It then
stops the existing `t2000.service` and
`t1000-3050.service` user services if installed, and starts the Docker studio. It
waits for the HTTP health check and prints the address. The models load when a
voice is first prepared or speech is requested. The initial engine is T-1000 MF2;
the studio selector still offers all four installed engines.

Only run one studio's synthesis on the 4 GB GPU at a time. The launch script
switches away from the native T-1000 studio to avoid retaining both models in
VRAM. Starting T-1000 should stop this container first. Docker restarts T-2000
after a reboot, with no synthesis model preloaded. Normal starts use the local
image and do not contact image registries. After changing application code or
dependencies, explicitly rebuild with `./start.sh --build`.

The configured server keeps `.cache/t2000-gpu.tar` as an intentional offline
recovery backup. Keep it when cleaning disposable caches: it allows `./start.sh`
to recover if a later Docker cleanup removes the stopped image. Existing images
are used as they are; the backup is loaded only when `t2000:gpu` is absent. The
archive contains application dependencies and code. Models, voice profiles,
recordings, and generated WAVs remain separate under the host data directories.

The equivalent Compose command is:

```sh
docker compose up -d --build --wait
```

Use `./start.sh` on the shared GPU host because it also releases the native
studios' GPU allocations. To stop or inspect the container:

```sh
docker compose stop studio
docker compose ps
docker compose logs --tail=100 studio
docker compose exec studio /opt/melo/bin/t2000 doctor
```

The app binds to `0.0.0.0` inside the container and publishes to host loopback
only. On the GPU laptop, open **http://127.0.0.1:7861**. From another machine:

```sh
ssh -N -L 7861:127.0.0.1:7861 prieyan@10.10.1.164
```

Then open **http://127.0.0.1:7861** locally. Microphone capture works through this
localhost connection. `T2000_PORT` in `.env` can change the published host port.

The container runs without root, with a read-only application filesystem and
writable persistent `./data` plus a named compiler/cache volume. Logs rotate at
30 MB. Removing/rebuilding the container preserves the host data. Models and
recordings never enter the Docker build context. Keep the native environments
until the Docker deployment has passed real synthesis checks.

## Bootstrap build recorded during setup

`Dockerfile.bootstrap` can reuse the already installed CPython 3.11 manylinux
dependencies on the configured x86_64 host. The regular `Dockerfile` remains the
installation route for a fresh machine. The recovery build copies dependency
files through two explicit named contexts, creates new container interpreters
and entrypoints, and installs only the missing F5 dependencies. It preserves the
T-1000 worker's pinned upstream commit metadata. Host interpreter paths, native
launch scripts, project models, recordings, and secrets are excluded.

The command below records the bootstrap used during setup, from the original
`/home/prieyan/T-2000` location. The checkout has since moved to
`/home/prieyan/weeb/T-2000`, and its native `.venv` has been removed after Docker
verification. Current offline recovery uses `cd ~/weeb/T-2000 && ./start.sh`,
which restores `.cache/t2000-gpu.tar` when the installed image is missing.
Repeating a bootstrap build requires the dependency directories in its named
contexts to exist.

Original bootstrap command:

```sh
docker buildx build -f Dockerfile.bootstrap -t t2000:gpu \
  --build-context melo-deps=/home/prieyan/T-2000/.venv/lib/python3.11/site-packages \
  --build-context t1000-deps=/home/prieyan/weeb/T-1000/.venv/lib/python3.11/site-packages \
  --output type=docker,dest=/home/prieyan/.local/state/tprojects-maintenance/2026-11-02/t2000-gpu.tar \
  .
docker load -i /home/prieyan/.local/state/tprojects-maintenance/2026-11-02/t2000-gpu.tar
mkdir -p .cache
mv /home/prieyan/.local/state/tprojects-maintenance/2026-11-02/t2000-gpu.tar .cache/t2000-gpu.tar
./start.sh
```

Verify actual CUDA imports and speech generation for all engines after loading;
reuse of host dependencies alone does not establish container compatibility.

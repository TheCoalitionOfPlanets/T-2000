# RTX 3050 validation — 2026-10-30

This is the original Melo/OpenVoice speed record. The laptop now runs the reference-conditioned F5 configuration documented in the [voice quality experiment](voice-quality-2026-10-30.md).

T-2000 is installed and running on `prieyan@10.10.8.211` in `/home/prieyan/T-2000`. MeloTTS/BERT and OpenVoice conversion both run on `cuda:0`; optional Nemotron recognition runs on CPU. The studio passed live upload, preparation, WebSocket speech generation, and WAV download through an SSH tunnel.

## Access and process control

On the GPU laptop, use **http://127.0.0.1:7861** directly. Firefox was opened to the studio and its window focused on the laptop desktop. The browser interface, synthesis models, and optional CPU recognition all run on that laptop.

The current local tunnel exposes the GPU studio at **http://127.0.0.1:7862** and forwards to remote loopback port 7861. To recreate the tunnel after it closes:

```sh
ssh -N -L 7862:127.0.0.1:7861 prieyan@10.10.8.211
```

The server runs as the user service `t2000.service`, configured in `/home/prieyan/.config/systemd/user/t2000.service`. It was started for this session; automatic startup at boot was not enabled.

```sh
ssh prieyan@10.10.8.211 'systemctl --user status t2000.service'
ssh prieyan@10.10.8.211 'systemctl --user restart t2000.service'
ssh prieyan@10.10.8.211 'systemctl --user stop t2000.service'
```

The unit uses the project's `.venv/bin/t2000 serve --host 127.0.0.1 --port 7861 --preload`, both device variables set to `cuda:0`, and four CPU threads. Stop this service before running an isolated benchmark, then restart it afterward.

## Environment and setup

| Component | Validated value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB |
| Driver | 615.78.08 |
| OS | Arch Linux, kernel 6.18.56-1-lts, x86_64 |
| Python | 3.11.15 in an isolated uv environment |
| PyTorch | 2.5.1+cu121, CUDA runtime 12.1 |
| Placement | Melo/BERT and converter on GPU; STT on CPU |
| Reference | T-1000 `assets/ref.wav`, mono PCM16, 24 kHz, 3.904 seconds |

The project was transferred over SSH. The supplied core and GPU requirement files were installed, and the pinned synthesis resources were downloaded on the GPU host. Offline integrity checking verified all **298 model/source/resource files**. The pinned Nemotron INT8 model was also downloaded and verified. The package environment is saved in [gpu-environment.txt](../data/validation/gpu-remote/gpu-environment.txt).

The host initially had about 8 GB of available disk. Installation exhausted most of it; pruning unused uv cache entries recovered 12.2 GiB, leaving about 12 GB available. Installed environments and project files were retained. The STT download archive was staged under `/tmp`.

## Completed checks

- CUDA availability and an actual GPU matrix operation passed.
- **52 Python tests passed**, Ruff passed, and browser AudioWorklet checks passed.
- CPU TTS with GPU conversion completed **24/24** timed requests.
- Both stages on GPU completed **24/24** timed requests in a subsequent run.
- All 48 benchmark WAVs were present, mono at 22,050 Hz, finite, nonempty, non-silent, and unclipped.
- Actual CPU STT loaded and transcribed the reference recording.
- The running studio served its HTML and reported both GPU devices. Upload, cached preparation, two-phrase generation, 64 ordered audio packets, and WAV download passed through the local tunnel. Packet frame totals and bytes agreed with the exported WAV.

The hybrid run was a functional check: early samples overlapped optional STT extraction and cache maintenance. Its timings should not be used for an isolated placement comparison. The subsequent both-GPU run began after setup completed, with no studio process running concurrently.

## Both-GPU benchmark

Eight English cases, three repeats each, one excluded warmup, randomized request order, one reference voice, seed 42, speed 1.0, and base speaker EN-US. Model loading and enrollment are outside the warm timing scope.

| Measurement | Median | p95 |
|---|---:|---:|
| Generation to final host PCM | 0.3611 s | 0.9372 s |
| First host PCM | 0.3457 s | 0.4775 s |
| Real-time factor | 0.0899 | 0.1123 |
| Generated audio duration | 4.1448 s | 13.8507 s |
| Base synthesis/resampling | 0.1304 s | 0.3873 s |
| Conversion/watermark | 0.2169 s | 0.5234 s |

Model loading took 5.00 seconds and conditioning preparation 0.16 seconds in the benchmark process. Peak CUDA allocation was **1120.28 MiB**, and peak reserved memory was **1250 MiB**. One-second `nvidia-smi` sampling observed up to 1343 MiB of total board usage during the sampling window; this includes other processes and can miss shorter peaks.

The studio's first live two-phrase request generated 6.33 seconds of audio in 2.48 seconds, with first host PCM at 1.97 seconds. A subsequent request generated 6.25 seconds in 0.95 seconds, with first host PCM at 0.53 seconds. These are separate observations from the warm benchmark; preloading alone does not guarantee the benchmark's median latency for an arbitrary first request.

## Artifacts and limits

- [Both-GPU benchmark summary](../data/benchmarks/gpu-remote/3050-both-gpu/REPORT.md), [raw JSON](../data/benchmarks/gpu-remote/3050-both-gpu/report.json), and [request CSV](../data/benchmarks/gpu-remote/3050-both-gpu/requests.csv).
- [Hybrid functional run](../data/benchmarks/gpu-remote/3050-cpu-tts-gpu-converter/REPORT.md).
- [WAV integrity, memory, and reference STT verification](../data/validation/gpu-remote/gpu-verification.json).
- [Live studio API verification](../data/validation/gpu-remote/api-smoke.json) and [sample WAV](../data/validation/gpu-remote/gpu-studio-sample.wav).
- [Generated-speech STT screen](../data/validation/gpu-remote/generated-stt-plain.json) and [earlier acronym screen](../data/validation/gpu-remote/generated-stt.json).

Recognition of the generated samples was not word-perfect: the plain sample's “is ready” was recognized as “was ready,” and an earlier sample's “GPU” was misrecognized. This screen cannot distinguish synthesis pronunciation differences from recognizer errors. Listen to the saved WAVs before judging intelligibility, pronunciation, identity, or naturalness. Short phrases also triggered upstream warnings that they were too short for watermark insertion.

Timing covers server generation to host PCM, excluding browser playback and audible device latency. CUDA allocator peaks exclude the driver and other processes. Twenty-four requests on one reference do not establish tail latency, broad speaker quality, or a matched improvement over T-1000. Browser playback gaps, audible latency, human listening, and a matched T-1000 comparison remain unmeasured.

Runtime artifacts are ignored by Git. Copies are stored locally under `data/validation/gpu-remote/` and `data/benchmarks/gpu-remote/`; original reports and generated WAVs remain under the remote project's `data/` directory.

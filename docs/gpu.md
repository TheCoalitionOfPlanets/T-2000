# RTX 3050 setup and SSH validation

The latest [own-voice delivery test](reference-delivery-2026-10-30.md) adds installed T-1000 SOAR/MF2 choices and a context-preserving controller. MF2 is now the laptop's selected starting option for listening, with the reviewed transcript and neutral reference boundary. This follows user feedback; the earlier F5 measurements below are retained as a historical experiment, not a perceptual winner.

An earlier laptop configuration selected **F5-TTS v1 at 32 steps** after the [voice experiment](voice-quality-2026-10-30.md). Its median generation time was 3.0539 seconds and median RTF 0.5582, with maximum RTF 1.2703. Subsequent listening feedback led to the reference-delivery work above. The portable configuration default remains Melo/OpenVoice; available F5 and T-1000 routes are alternatives for comparison.

Target: Linux x86_64, Python 3.11, RTX 3050 with 4 GB VRAM and a driver compatible with the pinned CUDA 12.1 PyTorch wheels. Both stages of the original Melo/OpenVoice cascade fit on the validated RTX 3050 Laptop GPU: an isolated 24/24-request run measured median RTF 0.0899, p95 RTF 0.1123, and median first host PCM at 0.3457 seconds. Peak allocator usage was 1120.28 MiB allocated and 1250 MiB reserved. See the [GPU validation record](gpu-validation-2026-10-30.md) for the environment and limits. When testing that cascade on a new system, initially keep Melo/BERT and STT on CPU, with the converter on GPU, then measure memory headroom.

## Transfer and install the application

Use your usual SSH connection. For example, from the local checkout, substitute the actual user and hostname:

```sh
rsync -av --exclude=.git --exclude=.venv --exclude=.venv-f5 --exclude=.cache --exclude=data \
  ./ user@gpu-host:~/T-2000/
ssh user@gpu-host
```

On the GPU machine:

```sh
cd ~/T-2000
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -c requirements/constraints.txt -e '.[dev,stt]'
.venv/bin/python scripts/fetch_stt.py
```

STT runs on CPU by default. Its transcript can be corrected before generation; F5 requires it to match the recording. A verified existing Nemotron directory can be reused through `STT_MODEL_DIR`.

## F5 reference-conditioned cloning

```sh
uv venv --python 3.11 .venv-f5
uv pip install --python .venv-f5/bin/python -r requirements/inference-f5.txt
.venv/bin/python scripts/setup_f5.py
.venv/bin/python scripts/setup_f5.py --check
export T2000_BACKEND=f5-tts-v1
export T2000_F5_PYTHON=.venv-f5/bin/python
export T2000_F5_STEPS=32
export T2000_TTS_DEVICE=cuda:0
export T2000_NUM_THREADS=4
.venv/bin/t2000 doctor
.venv/bin/t2000 serve --preload
```

The studio stays in `.venv`; the persistent inference process uses `.venv-f5`. Keep those environments separate. References must be complete 1–12 second clips; aim for 8–12 seconds of clean single-speaker speech. F5 outputs 24 kHz audio. Reference pace at 1.0× uses an estimate from recording and transcript lengths, without guaranteeing exact intonation, pitch or smoothness. F5 model pins and the CC-BY-NC-4.0 weight license are described in [model setup](models.md).

On the GPU laptop, open **http://127.0.0.1:7861** directly in its browser. All model and STT execution stays on that laptop. When connecting from another machine, use the SSH tunnel below. Run one synthesis service or isolated benchmark at a time on this GPU.

## Original fast Melo/OpenVoice backend

```sh
uv pip install --python .venv/bin/python -r requirements/inference-gpu.txt
.venv/bin/python scripts/setup_models.py
.venv/bin/python scripts/setup_models.py --check
```

Install the GPU profile instead of the CPU profile in `.venv` for this backend. The complete prepared model directory can also be copied from another machine; it needs source, tokenizers, NLP resources and manifest as well as speech checkpoints. See [models.md](models.md).

## Check and run

```sh
export T2000_BACKEND=melo-openvoice-v2
export T2000_TTS_DEVICE=cpu
export T2000_CONVERTER_DEVICE=cuda:0
export T2000_NUM_THREADS=4
nvidia-smi
.venv/bin/t2000 doctor
.venv/bin/pytest -q
.venv/bin/t2000 serve --preload
```

Returning to this backend requires setting the same variables in the service environment and restarting it. An already running service does not inherit a terminal's newly exported variables.

Preloading exposes dependency or memory errors before opening the studio. It does not establish steady-state performance. In a second **local** terminal:

```sh
ssh -N -L 7861:127.0.0.1:7861 user@gpu-host
```

Open **http://127.0.0.1:7861** locally. Microphone and playback stay local; model execution stays remote. Use the loopback binding. Stop the server before isolated benchmarks so another process does not duplicate weights in VRAM.

For the validated placement with both synthesis stages on GPU, set `T2000_TTS_DEVICE=cuda:0` before starting the process. STT stays on CPU. Real HTTP upload, voice preparation, WebSocket generation and WAV download passed through an SSH tunnel; browser audible latency and playback underruns still need measurement.

## Measure

Transfer a clean reference WAV to the remote machine:

```sh
T2000_TTS_DEVICE=cpu T2000_CONVERTER_DEVICE=cuda:0 \
  T2000_BACKEND=melo-openvoice-v2 .venv/bin/python scripts/benchmark.py \
  --reference /path/to/reference.wav --repeats 3 \
  --output-dir data/benchmarks/3050-first
```

Choose a new output directory each run. Repeat `--reference` for more speakers. The runner prepares references, excludes warmups, randomizes timed requests and records failures separately. A report is also saved on setup failure.

The report separates load/enrollment from warm generation, and records first host PCM, final PCM generation time, RTF, base/conversion times, audio duration, phrase gaps, and CUDA allocator peaks. Process RSS is a lifetime high-water mark. Allocator peaks exclude the driver, desktop and other processes; use `nvidia-smi` before and during the run to assess whole-board pressure.

The runner does not measure browser audible latency or assign quality scores. Check the studio for estimated audible latency and underruns. Listen to saved outputs for pronunciation, identity, rhythm, silence and phrase joins. Slower delivery or excess silence can improve RTF without improving useful speech generation. Early PCM is insufficient if later phrases cannot keep up.

Use the same references, novel texts, settings and completion rules for a fresh T-1000 comparison. Historical SOAR RTF 2.975 is context, not a matched baseline. Independent speaker evaluation and paired listening remain necessary before claiming equal quality.

The initial CPU-TTS/GPU-converter run completed 24/24 requests, but early samples overlapped optional STT setup, so it is a functional check rather than an isolated timing comparison. The subsequent run with both stages on GPU was isolated. On a new system, test both stages on GPU only if memory headroom permits. Change one placement setting per run and restart. Preserve outputs and failures.

For matched quality experiments, prepare the explicit CPU scorer/VAD models and use `scripts/compare_voice_quality.py` plus `scripts/benchmark_f5.py`; the [voice experiment](voice-quality-2026-10-30.md) gives commands and report links. Scoring runs after synthesis and does not enter generation timings.

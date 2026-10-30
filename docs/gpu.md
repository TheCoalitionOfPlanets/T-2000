# RTX 3050 setup and SSH validation

Target: Linux x86_64, Python 3.11, RTX 3050 with 4 GB VRAM and a driver compatible with the pinned CUDA 12.1 PyTorch wheels. Initially keep Melo/BERT and STT on CPU, with the converter on GPU. Fit and throughput must be measured.

## Transfer and install

Use your usual SSH connection. For example, from the local checkout, substitute the actual user and hostname:

```sh
rsync -av --exclude=.git --exclude=.venv --exclude=.cache --exclude=data \
  ./ user@gpu-host:~/T-2000/
ssh user@gpu-host
```

On the GPU machine:

```sh
cd ~/T-2000
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -c requirements/constraints.txt -e '.[dev,stt]'
uv pip install --python .venv/bin/python -r requirements/inference-gpu.txt
.venv/bin/python scripts/setup_models.py
.venv/bin/python scripts/setup_models.py --check
```

Install the GPU profile instead of the CPU profile. The complete prepared model directory can also be copied from another machine; it needs source, tokenizers, NLP resources and manifest as well as speech checkpoints. See [models.md](models.md). For optional CPU STT, run `scripts/fetch_stt.py` or point `STT_MODEL_DIR` at T-1000's verified Nemotron directory.

## Check and run

```sh
export T2000_TTS_DEVICE=cpu
export T2000_CONVERTER_DEVICE=cuda:0
export T2000_NUM_THREADS=4
nvidia-smi
.venv/bin/t2000 doctor
.venv/bin/pytest -q
.venv/bin/t2000 serve --preload
```

Preloading exposes dependency or memory errors before opening the studio. It does not establish steady-state performance. In a second **local** terminal:

```sh
ssh -N -L 7861:127.0.0.1:7861 user@gpu-host
```

Open **http://127.0.0.1:7861** locally. Microphone and playback stay local; model execution stays remote. Use the loopback binding. Stop the server before isolated benchmarks so another process does not duplicate weights in VRAM.

## Measure

Transfer a clean reference WAV to the remote machine:

```sh
T2000_TTS_DEVICE=cpu T2000_CONVERTER_DEVICE=cuda:0 \
  .venv/bin/python scripts/benchmark.py \
  --reference /path/to/reference.wav --repeats 3 \
  --output-dir data/benchmarks/3050-first
```

Choose a new output directory each run. Repeat `--reference` for more speakers. The runner prepares references, excludes warmups, randomizes timed requests and records failures separately. A report is also saved on setup failure.

The report separates load/enrollment from warm generation, and records first host PCM, final PCM generation time, RTF, base/conversion times, audio duration, phrase gaps, and CUDA allocator peaks. Process RSS is a lifetime high-water mark. Allocator peaks exclude the driver, desktop and other processes; use `nvidia-smi` before and during the run to assess whole-board pressure.

The runner does not measure browser audible latency or assign quality scores. Check the studio for estimated audible latency and underruns. Listen to saved outputs for pronunciation, identity, rhythm, silence and phrase joins. Slower delivery or excess silence can improve RTF without improving useful speech generation. Early PCM is insufficient if later phrases cannot keep up.

Use the same references, novel texts, settings and completion rules for a fresh T-1000 comparison. Historical SOAR RTF 2.975 is context, not a matched baseline. Independent speaker evaluation and paired listening remain necessary before claiming equal quality.

After the initial CPU-TTS/GPU-converter run, test both stages on GPU only if memory headroom permits. Change one placement setting per run and restart. Preserve outputs and failures.

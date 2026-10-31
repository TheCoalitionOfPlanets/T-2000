# Preparing local voice models

The application supports direct F5-TTS v1 cloning, reuse of installed T-1000 SOAR/MF2, and the original MeloTTS/OpenVoice V2 cascade. The portable default remains `melo-openvoice-v2`; the GPU laptop exposes its installed backends for comparison. Model setup is explicit and runtime inference uses local verified files. See the [reference delivery record](reference-delivery-2026-10-30.md) for subsequent user feedback and context-planning changes.

## F5-TTS v1 with isolated dependencies

Keep the application, CPU STT and evaluation tools in `.venv`. Install F5 synthesis dependencies separately:

```sh
uv venv --python 3.11 .venv-f5
uv pip install --python .venv-f5/bin/python -r requirements/inference-f5.txt
.venv/bin/python scripts/setup_f5.py --plan
.venv/bin/python scripts/setup_f5.py
.venv/bin/python scripts/setup_f5.py --check
.venv/bin/python scripts/fetch_stt.py
export T2000_BACKEND=f5-tts-v1
export T2000_F5_PYTHON=.venv-f5/bin/python
export T2000_F5_STEPS=32
export T2000_TTS_DEVICE=cuda:0
.venv/bin/t2000 serve --preload
```

The persistent worker runs in `.venv-f5`; the studio runs in `.venv`. The interpreter path retains the virtual environment symlink rather than resolving to the system interpreter. No inference request installs packages or downloads models. Source exports are limited to inference classes to avoid importing the upstream trainer; model kernels are unchanged. The pinned setup verifies both extracted source bytes and model resources and records them under `data/models/f5/manifest.json`.

| F5 resource | Revision |
|---|---|
| F5 source | `45868abaabd74aabf03cd673d6dd00cc9d9c01ae` |
| F5TTS_v1_Base weights and vocabulary | `84e5a410d9cead4de2f847e7c9369a6440bdfaca` |
| Vocos 24 kHz weights and config | `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21` |

F5 accepts complete 1–12 second references and requires the exact transcript. The studio suggests 8–12 seconds of clean single-speaker speech; the measured reference set was shorter and does not validate all recording conditions. CPU STT is enabled by default, remains editable, and automatic writes cannot replace a correction saved while recognition runs. Missing or incorrect reference words should be corrected before generation.

The worker derives a 24 kHz mono conditioning branch while retaining the original recording. It follows the pinned upstream edge treatment: -42 dBFS boundaries in 10 ms steps and a 50 ms tail pad. Longer references are rejected rather than silently cropped. A duration heuristic based on reference audio and text lengths estimates pace; 1.0× is not a promise of identical timing or intonation. Completed audio receives multiplicative attenuation when needed to leave a 0.98 peak margin. `raw_peak`, `raw_clipped_fraction` and `output_gain` remain in worker metrics; safe exported PCM does not prove the unscaled model output stayed below full scale.

Output is native 24 kHz complete-phrase audio. F5 uses more generation time than the cascade, with some measured requests slower than playback. Speaker identity, pitch, emotion and smoothness still need listening evaluation. F5 code is MIT and its pretrained weights are **CC-BY-NC-4.0**. [Pinned inference implementation](https://github.com/SWivid/F5-TTS/blob/45868abaabd74aabf03cd673d6dd00cc9d9c01ae/src/f5_tts/infer/utils_infer.py), [weight model card](https://huggingface.co/SWivid/F5-TTS/tree/84e5a410d9cead4de2f847e7c9369a6440bdfaca).

## MeloTTS/OpenVoice V2

This backend uses local MeloTTS speech generation followed by OpenVoice V2 tone-color conversion. Set `T2000_BACKEND=melo-openvoice-v2` and restart to select it. Starting the web app, viewing diagnostics, and importing the backend do not download models. Missing models produce an actionable error; there is no substitute voice or fabricated speech fallback.

Real CPU inference completed two requests with outbound sockets blocked and no connection attempts recorded. Both outputs contained finite, unclipped audio; the [CPU smoke record](../data/validation/cpu-smoke.json) captures the results. This verifies functional inference and offline model reuse. Subsequent [RTX 3050 validation](gpu-validation-2026-10-30.md) verified the 298 synthesis files and completed 24/24 isolated requests with both stages on GPU, peaking at 1120.28 MiB allocated and 1250 MiB reserved. Human evaluation of speaker identity and naturalness, and browser audible latency, remain pending.

Use Python 3.11 on Linux x86_64. From the repository, create the environment and install the editable core as described in the README. Choose one inference dependency file:

```sh
uv pip install --python .venv/bin/python -r requirements/inference-cpu.txt
```

For the RTX 3050, use the CUDA 12.1 PyTorch build instead:

```sh
uv pip install --python .venv/bin/python -r requirements/inference-gpu.txt
```

These files select Torch/torchaudio and apply `constraints.txt`, a snapshot of shared dependencies from the Linux/Python 3.11 CPU environment. CUDA-specific dependencies are selected by the pinned Torch GPU wheel. The GPU profile passed on Arch Linux 6.18.56-1-lts with Python 3.11.15, Torch 2.5.1+cu121, NVIDIA driver 615.78.08, and an RTX 3050 Laptop GPU with 4 GB VRAM. Setuptools 75.8.0 retains `pkg_resources` for the legacy PyKakasi frontend. Setup avoids Melo's upstream `setup.py`, whose install hook downloads a full UniDic dictionary. The prepared source is imported directly from the model directory. MeCab, IPADIC and UniDic Lite satisfy tokenizer imports; full UniDic is not required for English. A separately installed full `unidic` without its dictionary may need correcting in that environment.

Inspect the download plan without making a network request:

```sh
.venv/bin/python scripts/setup_models.py --plan
```

Download and verify the pinned sources, weights, tokenizers, and NLP resources:

```sh
.venv/bin/python scripts/setup_models.py --model-dir data/models
.venv/bin/python scripts/setup_models.py --model-dir data/models --check
```

Allow roughly 0.9 GB of downloads and several GB of free disk space including the inference environment. Setup records SHA-256 for every prepared resource in `manifest.json`, verifies weight checksums against the pinned model metadata, and verifies Git blob hashes for small Hugging Face files. Source archives and NLTK resources use immutable repository revisions. NLTK ZIPs additionally have pinned Git blob checksums and sizes in the setup script; fetching tries GitHub's blob/contents APIs before a verified raw fallback, avoiding dependence on the raw CDN alone. Downloads finish into temporary files before replacing destinations; an interrupted setup does not publish a complete manifest. Rerunning reuses complete files whose checksums match; interrupted partial downloads start again. `--check` is offline and does not import PyTorch.

The principal pins are:

| Resource | Revision |
|---|---|
| MeloTTS source | `209145371cff8fc3bd60d7be902ea69cbdb7965a` |
| OpenVoice source | `74a1d147b17a8c3092dd5430504bd83ef6c7eb23` |
| MeloTTS-English weights/config | `bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef` |
| OpenVoice V2 weights/config/source embeddings | `f36e7edfe1684461a8343844af60babc2efbb727` |
| WavMark weights | `0ad3c7b74f641bddb61f6b85cdf2de0d93a5bfef` |

The exact frontend and NLTK revisions are printed by `--plan` and retained in the manifest. English BERT weights are included. Upstream imports tokenizers for several other languages even when synthesizing English, so those tokenizers are also prepared; their large speech/BERT model weights are not downloaded. This backend currently accepts `EN` only.

The directory layout is:

```text
data/models/
  manifest.json
  sources/MeloTTS/melo/
  sources/OpenVoice/openvoice/
  melo/config.json
  melo/checkpoint.pth
  openvoice/converter/config.json
  openvoice/converter/checkpoint.pth
  openvoice/base_speakers/ses/en-*.pth
  watermark/model.pkl
  hf/
  nltk_data/
  downloads/
```

At runtime, Hugging Face is put in offline mode and the backend checks local NLTK resources before importing the phonemizer. Setup retains both verified ZIP files and extracted resources in `nltk_data`: g2p_en's legacy import checks require the literal `cmudict.zip` and `averaged_perceptron_tagger.zip` paths. Checkpoints and cached conditioning use explicit `weights_only=True` loading. The adapter constructs the Melo network with the same parameters as its upstream API to avoid changing `torch.load` globally. It initializes OpenVoice through its base initializer and then attaches the local watermark model, avoiding a keyword-forwarding bug in the pinned convenience initializer. The one source adjustment during setup makes Melo read its pinned CMU dictionary rather than an untracked pickle cache; the adjusted source checksum is in the manifest.

Initial CPU settings are:

```sh
export T2000_TTS_DEVICE=cpu
export T2000_CONVERTER_DEVICE=cpu
export T2000_NUM_THREADS=4
export T2000_SPEAKER=EN-US
```

For the first GPU experiment, keep Melo/BERT on CPU and move the converter:

```sh
export T2000_TTS_DEVICE=cpu
export T2000_CONVERTER_DEVICE=cuda:0
```

Both stages also fit on the validated RTX 3050. Set `T2000_TTS_DEVICE=cuda:0` as well to use that placement; see the [GPU guide](gpu.md) before benchmarking a different machine.

Restart the process after changing placement, source files, dependencies, or base voice. The conditioning signature includes actual file hashes, runtime versions, model settings, and base speaker. Conditioning is regenerated under a new signature while the original reference profile can be reused. Source embeddings are selected to match Melo's actual base speaker; English base choices are `EN-US`, `EN-BR`, `EN-INDIA`, `EN-AU`, and `EN-DEFAULT`.

References retain their uploaded sample rate in the application. Enrollment makes an independent high-quality resampled branch at the converter's 22,050 Hz rate and extracts the target embedding once. The default `full` reference mode uses the complete selected recording. The optional `speech` mode uses explicitly prepared Silero VAD, padded speech intervals and averaged segment embeddings; it is an experimental candidate, not a general improvement. The target tensor is saved atomically on CPU and moved to the selected converter device when used. Melo's 44,100 Hz output is resampled in memory before conversion, avoiding an intermediate generated WAV.

Conversion operates on complete phrases. Phrase output permits incremental playback but does not establish frame-level streaming or a first-audible latency bound. The adapter reports separate base/resampling and conversion/watermark times. CUDA timing synchronizes the selected device. The optional upstream watermark stays enabled with its `@MyShell` message and sample-block behavior; very short phrases may be too short for all watermark blocks. No claim is made that every short output carries a recoverable mark.

The selected tone-color converter leaves accent and delivery largely with the base TTS. Speed is adjustable; arbitrary emotion transfer from a recording is not implemented. Seeded synthesis supports repeatable experiments within an environment, without promising bit-identical output across devices or runtimes.

Primary implementation references: [Melo API](https://github.com/myshell-ai/MeloTTS/blob/209145371cff8fc3bd60d7be902ea69cbdb7965a/melo/api.py), [official OpenVoice V2 pairing](https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/demo_part3.ipynb), [OpenVoice converter API](https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/openvoice/api.py), and [tone-color limitations](https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/docs/QA.md).

## Optional VAD and independent evaluation models

The delivery controller also uses the verified CPU VAD to cache speech duration, pause and pace diagnostics. These measurements do not infer accent or emotion.

```sh
.venv/bin/python scripts/setup_quality_models.py --plan
.venv/bin/python scripts/setup_quality_models.py
.venv/bin/python scripts/setup_quality_models.py --check
```

This explicitly prepares `quality/silero_vad.onnx` and `quality/wespeaker_en_voxceleb_resnet34.onnx`, with pinned byte sizes and SHA-256 checks. Runtime checks are offline. Both run on CPU through the existing sherpa-onnx dependency, avoiding extra GPU allocations while scoring. The WeSpeaker model is independent of both synthesis conditioning encoders; its cosine score has no calibrated likeness percentage or pass threshold. See the [evaluation method and artifacts](voice-quality-2026-10-30.md).

## Reuse an installed T-1000 runtime

No second copy of T-1000's models or dependencies is needed on the GPU laptop:

```sh
export T2000_T1000_ROOT=/home/prieyan/weeb/T-1000
export T2000_T1000_PYTHON=/home/prieyan/weeb/T-1000/.venv/bin/python
export T2000_TTS_DEVICE=cuda:0
export T2000_BACKEND=t1000-mf2  # or t1000-soar
.venv/bin/t2000 serve --preload
```

The adapter verifies the pinned existing checkpoint and runtime, and starts a persistent isolated worker. SOAR uses ten flow steps and CFG 1.2; MF2 retains its trained two-step sampling settings. Complete reference audio and the reviewed transcript form the acoustic continuation context. Native output is 48 kHz, with no pitch or speed DSP. Reference limits and GPU fit use the existing low-VRAM profile; unload the other synthesis worker before loading either model. Cold loading can be slow under host RAM/swap pressure. Compare the outputs against the actual recording rather than selecting a winner by speaker similarity alone.

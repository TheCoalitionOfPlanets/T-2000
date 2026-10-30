# Preparing the MeloTTS/OpenVoice V2 backend

The application uses real local MeloTTS speech generation followed by OpenVoice V2 tone-color conversion. The model setup is a separate, explicit command. Starting the web app, viewing diagnostics, and importing the backend do not download models. Missing models produce an actionable error; there is no substitute voice or fabricated speech fallback.

Real CPU inference completed two requests with outbound sockets blocked and no connection attempts recorded. Both outputs contained finite, unclipped audio; the [CPU smoke record](../data/validation/cpu-smoke.json) captures the results. This verifies functional inference and offline model reuse. RTX 3050 memory fit, GPU latency, and human evaluation of speaker identity and naturalness remain pending; these two CPU requests do not establish a performance promise.

Use Python 3.11 on Linux x86_64. From the repository, create the environment and install the editable core as described in the README. Choose one inference dependency file:

```sh
uv pip install --python .venv/bin/python -r requirements/inference-cpu.txt
```

For the RTX 3050, use the CUDA 12.1 PyTorch build instead:

```sh
uv pip install --python .venv/bin/python -r requirements/inference-gpu.txt
```

These files select Torch/torchaudio and apply `constraints.txt`, a snapshot of shared dependencies from the Linux/Python 3.11 CPU environment. CUDA-specific dependencies are selected by the pinned Torch GPU wheel; the GPU environment is still untested. Setuptools 75.8.0 retains `pkg_resources` for the legacy PyKakasi frontend. Setup avoids Melo's upstream `setup.py`, whose install hook downloads a full UniDic dictionary. The prepared source is imported directly from the model directory. MeCab, IPADIC and UniDic Lite satisfy tokenizer imports; full UniDic is not required for English. A separately installed full `unidic` without its dictionary may need correcting in that environment.

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

Restart the process after changing placement, source files, dependencies, or base voice. The conditioning signature includes actual file hashes, runtime versions, model settings, and base speaker. Conditioning is regenerated under a new signature while the original reference profile can be reused. Source embeddings are selected to match Melo's actual base speaker; English base choices are `EN-US`, `EN-BR`, `EN-INDIA`, `EN-AU`, and `EN-DEFAULT`.

References retain their uploaded sample rate in the application. Enrollment makes an independent high-quality resampled branch at the converter's 22,050 Hz rate and extracts the target embedding once. The backend uses the complete selected recording, without downloading a separate VAD or ASR model: choose a clean single-speaker clip with little silence. The target tensor is saved atomically on CPU and moved to the selected converter device when used. Melo's 44,100 Hz output is resampled in memory before conversion, avoiding an intermediate generated WAV.

Conversion operates on complete phrases. Phrase output permits incremental playback but does not establish frame-level streaming or a first-audible latency bound. The adapter reports separate base/resampling and conversion/watermark times. CUDA timing synchronizes the selected device. The optional upstream watermark stays enabled with its `@MyShell` message and sample-block behavior; very short phrases may be too short for all watermark blocks. No claim is made that every short output carries a recoverable mark.

The selected tone-color converter leaves accent and delivery largely with the base TTS. Speed is adjustable; arbitrary emotion transfer from a recording is not implemented. Seeded synthesis supports repeatable experiments within an environment, without promising bit-identical output across devices or runtimes.

Primary implementation references: [Melo API](https://github.com/myshell-ai/MeloTTS/blob/209145371cff8fc3bd60d7be902ea69cbdb7965a/melo/api.py), [official OpenVoice V2 pairing](https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/demo_part3.ipynb), [OpenVoice converter API](https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/openvoice/api.py), and [tone-color limitations](https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/docs/QA.md).

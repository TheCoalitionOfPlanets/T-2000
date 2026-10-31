# Reference voice experiment — 2026-10-30

This is the earlier F5 experiment. Subsequent user listening feedback led to the [own-voice delivery and reference-boundary work](reference-delivery-2026-10-30.md), with installed-engine comparison and MF2 selected for further listening. The measurements below remain historical proxies.

The GPU laptop now selects F5-TTS v1 at 32 sampling steps to prioritize reference voice likeness over the original cascade's generation speed. On a small matched set of new texts, F5 improved independent speaker-similarity and ASR proxies. This is a measured configuration choice for this laptop, not an exact-cloning claim or a general model ranking. The portable code default remains `melo-openvoice-v2`.

## Scope and method

Experiments ran on the RTX 3050 Laptop GPU with 4 GB VRAM, Arch Linux 6.18.56-1-lts, NVIDIA driver 615.78.08 and Python 3.11.15. Both inference environments used Torch 2.5.1+cu121. The application, Nemotron CPU STT and independent evaluation ran in `.venv`; the persistent F5 worker used `.venv-f5`.

The initial reference set contained four saved reference recordings shorter than 6.3 seconds. Two texts and seed 42 were used for the configuration experiments. The later check used four new texts (`short`, `names`, `contrast`, `paragraph`) and seed 43 for all four references, giving 16 exact control/candidate pairs. These are held-out texts, not held-out speakers; the recordings do not establish four distinct speakers. New texts are held out from this project's parameter choice; their presence in model training data is unknown.

Pairs match the reference SHA-256, case ID, text and seed. The original control is `EN-US/full/tau=0.3`. Failed, incomplete, clipped or unscoreable outputs do not enter eligible similarity/timing medians. Scoreable ASR errors are retained separately, including when another scorer fails. Reports preserve failure counts, frame counts, duration changes, hashes and model provenance. Model loading, reference preparation and later CPU scoring are separate from warm generation timings.

Speaker similarity uses a pinned **WeSpeaker English VoxCeleb ResNet34 ONNX** model through sherpa-onnx on CPU. It is independent of both synthesis conditioning encoders. Every reference/output is scored as a complete mono clip, resampled with soxr HQ to 16 kHz, with L2-normalized embeddings and cosine comparison. There is no random crop, scoring-time silence removal, likeness percentage or calibrated pass threshold. [Official speaker API](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speaker-identification.py).

Nemotron INT8 ASR supplies full-file transcripts. WER/CER use deterministic edit counts after NFKC, case folding, punctuation/symbol removal and whitespace collapse; numbers are not expanded to hide differences. ASR agreement is not human-rated intelligibility. Approximate autocorrelation F0 median/IQR, adjacent pitch-step diagnostics, spectrum and recognized words per full-duration second describe pitch, spectral shape and pace. They are not identity, naturalness or smoothness scores.

## Results

The original Melo/OpenVoice sweep covered five correctly paired English base speakers and two reference modes, at tau 0.3: **4 references × 2 texts × 10 configurations = 80 outputs**. Speech-only conditioning uses verified Silero VAD and averaged reference-segment embeddings. Gains varied by reference and sometimes came with more ASR errors. Neither a new base speaker nor speech-only conditioning was automatically selected as a universal improvement. OpenVoice's tau scales posterior noise; it is not a cloning-strength control. [Pinned converter source](https://github.com/myshell-ai/OpenVoice/blob/74a1d147b17a8c3092dd5430504bd83ef6c7eb23/openvoice/models.py).

The first raw-reference F5 attempt produced **17 word errors across 128 expected words** and two clipped outputs that were excluded from eligible results. Those failures remain recorded. The tested correction followed the pinned upstream reference edge treatment (-42 dBFS detection in 10 ms steps, followed by 50 ms of tail silence). Original reference recordings are retained. F5 output now receives a single multiplicative gain reduction when needed to leave a 0.98 peak margin; raw peak, raw full-scale fraction and output gain are recorded. Safe exported PCM does not imply the unscaled model output was below full scale. [Pinned upstream preparation](https://github.com/SWivid/F5-TTS/blob/45868abaabd74aabf03cd673d6dd00cc9d9c01ae/src/f5_tts/infer/utils_infer.py).

| Measurement | Initial prepared set: Melo | Initial prepared set: F5 | New-text set: Melo | New-text set: F5 |
|---|---:|---:|---:|---:|
| Eligible / requests | 8/8 | 8/8 | 16/16 | 16/16 |
| ASR word errors / expected words | 1/128 | 1/128 | 15/256 | 4/256 |
| ASR WER | 0.78125% | 0.78125% | 5.859375% | 1.5625% |
| Median independent speaker cosine | 0.630974 | 0.812942 | 0.644123 | 0.807571 |
| Median generation seconds | 0.4320 | 3.7327 | 0.3573 | 3.0539 |
| Median RTF | 0.0766 | 0.5046 | 0.0857 | 0.5582 |

The median **paired** cosine delta was **+0.07400** on the initial prepared set and **+0.10933996** on the new-text set. A median of pairwise differences is different from subtracting two aggregate medians. Cosines are model outputs, not percentages of voice likeness.

All 16 held-out F5 WAVs were finite, nonempty, mono 24 kHz and eligible under the output clipping checks. Maximum F5 RTF was **1.2703** and maximum CUDA reserved memory was **866 MiB**. Allocator memory excludes the driver and other processes. Some requests generated more slowly than playback. The benchmark's first host PCM is a complete phrase; its median equals generation time for the F5 worker and does not measure browser audible latency.

Melo timings come from the engine's warm worker-to-final-PCM interval. F5 timings in this experiment come from the isolated worker's inference and chunk assembly interval, excluding reference preparation, IPC and final WAV export. They describe the measured generation paths, not a matched browser or full-request latency test.

Output duration differed by more than ten percent in **7/8 initial** and **14/16 held-out** F5 pairs. Pace differences and longer outputs affect RTF and embedding measurements, so a higher cosine alone is insufficient for declaring a better result. Review the actual words, duration and paired audio.

## Reference pace and studio behavior

The studio adopts T-1000's layout and keeps recording, saved voices, editable transcripts, diagnostics and phrase playback together. Saving a reference transcribes it with CPU STT by default. Manual corrections are preserved by guarded metadata updates, including when recognition completes after a correction. F5 preparation can transcribe an older empty profile, then requires a nonempty matching transcript.

F5 conditions directly on reference audio and words. Its duration estimate uses reference frame count divided by transcript byte length, scaled by new text length and the pace multiplier. **Reference pace at 1.0× is an estimate**, not an alignment that copies exact timing, pitch movement, accent, emotion or smoothness. The existing plots and report diagnostics support inspection; they do not guarantee those qualities.

F5 accepts complete **1–12 second** references and the UI suggests **8–12 seconds** of clean, single-speaker speech. Clips over the limit are rejected rather than truncated. The measured short-reference set does not establish that every reference in the accepted range works equally well. Longer or noisier references, new speakers, other accents and new texts need more evaluation.

## Reproduce and inspect

From an application environment installed as described in the README:

```sh
uv venv --python 3.11 .venv-f5
uv pip install --python .venv-f5/bin/python -r requirements/inference-f5.txt
.venv/bin/python scripts/setup_f5.py
.venv/bin/python scripts/setup_f5.py --check
.venv/bin/python scripts/fetch_stt.py
.venv/bin/python scripts/setup_quality_models.py
.venv/bin/python scripts/setup_quality_models.py --check
```

The pinned F5 source revision is `45868abaabd74aabf03cd673d6dd00cc9d9c01ae`; F5TTS_v1_Base model revision is `84e5a410d9cead4de2f847e7c9369a6440bdfaca`; Vocos revision is `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`. The model manifest records SHA-256 values. F5 code is MIT; pretrained F5 weights are **CC-BY-NC-4.0**. [Official model card](https://huggingface.co/SWivid/F5-TTS/tree/84e5a410d9cead4de2f847e7c9369a6440bdfaca), [pinned source](https://github.com/SWivid/F5-TTS/tree/45868abaabd74aabf03cd673d6dd00cc9d9c01ae).

Stop the studio before an isolated GPU experiment. Use fresh output directories:

```sh
T2000_BACKEND=melo-openvoice-v2 T2000_TTS_DEVICE=cuda:0 \
  T2000_CONVERTER_DEVICE=cuda:0 .venv/bin/python scripts/compare_voice_quality.py \
  --all-profiles --cases benchmarks/voice-quality.json \
  --speakers EN-US EN-BR EN-INDIA EN-AU EN-DEFAULT \
  --reference-modes full speech --taus 0.3 --seeds 42 \
  --output-dir data/benchmarks/quality-repeat
.venv/bin/python scripts/benchmark_f5.py \
  --baseline-report data/benchmarks/quality-repeat/report.json \
  --steps 32 --reference-preprocessing upstream-edges \
  --output-dir data/benchmarks/f5-repeat32
```

To test new texts, supply a new cases JSON file to the control harness and use the same seed for its F5 comparison. Inference runs before CPU ASR and embedding scoring. The HTML reports embed reference/candidate WAVs for local listening; they contain recordings and should be handled as private local artifacts.

| Run | Machine-readable report | Listening page |
|---|---|---|
| 80-output Melo matrix | [quality-first/report.json](../data/benchmarks/quality-first/report.json) | [quality-first/LISTEN.html](../data/benchmarks/quality-first/LISTEN.html) |
| Prepared F5, initial texts | [f5-prepared32/report.json](../data/benchmarks/f5-prepared32/report.json) | [f5-prepared32/LISTEN.html](../data/benchmarks/f5-prepared32/LISTEN.html) |
| Held-out Melo control | [quality-heldout/report.json](../data/benchmarks/quality-heldout/report.json) | [quality-heldout/LISTEN.html](../data/benchmarks/quality-heldout/LISTEN.html) |
| Held-out F5 comparison | [f5-heldout32/report.json](../data/benchmarks/f5-heldout32/report.json) | [f5-heldout32/LISTEN.html](../data/benchmarks/f5-heldout32/LISTEN.html) |

These runtime artifacts are ignored by Git. The same run directories remain under `/home/prieyan/T-2000/data/benchmarks/` on the GPU laptop; a source-only checkout does not include them.

## Deployment and remaining checks

The GPU laptop's service drop-in at `/home/prieyan/.config/systemd/user/t2000.service.d/quality.conf` selects `T2000_BACKEND=f5-tts-v1`, an absolute `.venv-f5/bin/python` path and `T2000_F5_STEPS=32`. The base unit sets `T2000_TTS_DEVICE=cuda:0`. A live status request confirmed the F5 worker loaded on CUDA, STT enabled and available on CPU, and the 12-second reference limit. The portable default remains Melo/OpenVoice; switching back requires changing `T2000_BACKEND` in this drop-in to `melo-openvoice-v2`, running `systemctl --user daemon-reload`, and restarting `t2000.service`.

The integration has passed **129 Python tests, Ruff and AudioWorklet checks locally and on the remote laptop**. Live Chromium verification against the deployed GPU service also passed reference preparation, automatic filling of an older empty transcript, generation, AudioWorklet playback, WAV download and mobile layout. No page errors or horizontal mobile overflow occurred. The request generated 7.57 seconds of 24 kHz audio in 4.58 seconds, with browser-estimated first audible time of 4.77 seconds and zero reported playback gaps. These are one-request observations, not device-measured audible timing or a sustained underrun guarantee. A separate real upload with the transcription flag omitted confirmed default STT and preserved the existing profile ID.

The studio and the paired listening page were opened in Firefox on the GPU laptop. A separate live pace-control check kept reference, text and seed fixed: the 0.85× request produced 6.49 seconds of audio and 1.15× produced 4.79 seconds. This checks the multiplier; it does not establish exact imitation of the speaker's rhythm.

Live artifacts are [browser-smoke.json](../data/validation/quality-live/browser-smoke.json), [pace-smoke.json](../data/validation/quality-live/pace-smoke.json), [desktop view](../data/validation/quality-live/desktop.png), [mobile view](../data/validation/quality-live/mobile.png), and [generated WAV](../data/validation/quality-live/quality-studio-sample.wav). Human listening for identity, natural delivery, phrase joins and smoothness, device-measured audible latency, longer playback runs, more speakers, and a matched T-1000 model comparison remain outstanding.

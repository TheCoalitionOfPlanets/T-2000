# Implementation validation

The latest [reference-delivery record](reference-delivery-2026-10-30.md) covers the user's own recording, prompt-word leakage, reviewed transcript and boundary correction, direct T-1000 reuse, fast context planning, and live MF2 integration. The full local suite now has **192 passing tests**; earlier counts below describe their historical stages.

The initial checks below record CPU and implementation validation. Subsequent RTX 3050 results are recorded separately below and in the [GPU validation report](gpu-validation-2026-10-30.md). The later [voice experiment](voice-quality-2026-10-30.md) adds independent speaker-similarity and ASR measurements. Speaker quality still needs human evaluation.

- **52 Python tests passed**, covering audio/reference handling, cache invalidation, optional STT, real-adapter boundaries, packet ordering, cancellation, concurrency and publication failures.
- Ruff passed. The wheel built successfully and includes the browser assets.
- AudioWorklet checks passed for capture finalization, resampling, packet order, gap counting and bounded memory.
- Chromium integration passed with an injected test backend for upload, microphone capture, transcript editing, preparation, playback, download and mobile layout. These test signals/timings are not synthesized speech measurements.
- Explicit model setup completed and offline verification passed for **298 pinned model/source/resource files**.
- Real CPU synthesis produced two complete 22,050 Hz WAVs using T-1000's bundled reference. Outbound IPv4/IPv6 socket connections were blocked for this check; no connection attempts were observed. Both outputs were finite, non-silent and unclipped.
- The reused Nemotron recognizer transcribed the two outputs to the intended words, ignoring punctuation. This is a small intelligibility screen, not human-rated naturalness or identity evaluation.
- The benchmark runner completed a two-case CPU smoke run, including one multi-phrase request. Both timed requests completed and JSON/CSV/Markdown outputs were written. This small run does not establish tail latency or a speedup over T-1000.
- Chromium also completed a real CPU-backed generation, AudioWorklet playback and WAV download with no page errors. Audible latency remains a browser estimate, not a measured device/microphone result.

Local generated records are under `data/validation/cpu-smoke.json` and `data/benchmarks/cpu-smoke/`; WAVs and request sidecars are under `data/output/`. Runtime artifacts are ignored by Git and are not part of a source-only transfer. Tests use Python 3.11 and the shared dependency versions in `requirements/constraints.txt`; CPU inference uses the pinned CPU Torch wheel.

The real CPU run exposed and resolved three compatibility issues: legacy PyKakasi needs a setuptools release with `pkg_resources`; g2p_en checks for literal NLTK ZIP resources as well as extracted data; and the pinned OpenVoice convenience constructor forwards an unsupported watermark keyword. Setup now prepares those resources and the adapter initializes the same converter base network before attaching its verified local watermark model.

## RTX 3050 validation, 2026-10-30

The SSH system ran Arch Linux 6.18.56-1-lts, Python 3.11.15, Torch 2.5.1+cu121 and NVIDIA driver 615.78.08 on an RTX 3050 Laptop GPU with 4 GB VRAM.

- **52 Python tests, Ruff and browser audio checks passed** on the GPU system.
- Offline verification passed for **298 synthesis files**. Nemotron STT was installed and actual reference transcription passed.
- CPU TTS with GPU conversion completed **24/24 requests** as a functional check. Early samples overlapped STT setup, so these timings are not an isolated comparison.
- Both synthesis stages on GPU completed **24/24 isolated warm requests**. Median generation time was **0.3611 seconds**, median first host PCM **0.3457 seconds**, median RTF **0.0899**, and p95 RTF **0.1123**. Peak CUDA allocator usage was **1120.28 MiB allocated** and **1250 MiB reserved**; this excludes driver and other-process memory.
- The GPU studio passed real HTTP upload, voice preparation, WebSocket generation and WAV download through an SSH tunnel.

See the [GPU validation report](gpu-validation-2026-10-30.md) for artifacts and measurement limits. Human listening for identity/naturalness, browser audible latency and playback underruns, and a matched T-1000 comparison remain pending. See [the GPU guide](gpu.md) to repeat the run.

## Reference likeness and F5 integration, 2026-10-30

- **129 Python tests, Ruff and AudioWorklet checks passed locally and on the GPU system** after adding the direct backend, default STT and evaluation tools. The earlier 52-test count above is the historical initial implementation record.
- The T-1000 studio layout was adopted. Reference STT is enabled by default, explicit transcripts remain editable, and guarded metadata updates preserve corrections saved while ASR is pending. F5 displays reference pace guidance and the actual 24 kHz output rate.
- A matched Melo/OpenVoice sweep generated and scored **80 outputs**: four references, two texts and ten speaker/conditioning configurations. Results varied by reference; speech-only conditioning did not justify a blanket default change.
- An initial raw-reference F5 trial had **17/128 ASR word errors** and **two clipped outputs excluded from eligible results**. It was retained as a failed experiment rather than used for promotion.
- After upstream-aligned edge preparation and recorded linear peak attenuation, 32-step F5 completed **8/8 eligible initial requests**, with **1/128 word errors** for each backend and median paired speaker-cosine delta **+0.0740**.
- On four new text cases at seed 43, each repeated for the same four references, F5 completed **16/16 eligible 24 kHz outputs**. ASR counted **4/256 word errors (1.5625%)**, versus **15/256 (5.8594%)** for the matched Melo control. Median paired independent speaker-cosine delta was **+0.10934**. These scores are proxies, not calibrated identity percentages.
- Held-out F5 median generation time was **3.0539 seconds** versus **0.3573 seconds** for Melo. Median F5 RTF was **0.5582**, maximum **1.2703**, and maximum CUDA reserved memory **866 MiB**. Some requests generated more slowly than playback.
- The laptop service selects F5 with 32 steps; the portable application default remains Melo/OpenVoice. A status check confirmed the F5 service loaded on CUDA with CPU STT available and a 12-second reference limit.

The four reused reference recordings were all under 6.3 seconds; held-out texts do not constitute held-out speakers. Pitch, spectral shape and transcript-derived pace are descriptive diagnostics. Exact identity, intonation, emotion and smoothness are not established by these results. Live Chromium verification against the deployed F5 service passed preparation, automatic transcript filling, speech generation, playback, WAV download and mobile layout with no page errors. The observed request reported zero playback gaps; audible timing remains a browser estimate. See the [full voice report and listening artifacts](voice-quality-2026-10-30.md).

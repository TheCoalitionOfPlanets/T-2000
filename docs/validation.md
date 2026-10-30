# Initial implementation validation

This records implementation checks, not an RTX 3050 performance or speaker-quality result.

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

Still to validate on the SSH machine: CUDA installation, 4 GB GPU memory fit, sustained throughput, first audible latency, playback underruns, a matched T-1000 comparison, and human listening for identity/naturalness. See [the GPU guide](gpu.md).

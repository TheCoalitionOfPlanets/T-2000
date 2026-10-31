# Own-voice delivery and reference boundaries — 2026-10-30

The listening target is the user's own voice, especially accent, sentence flow, naturalness and emotion. T-1000 is a working comparison, not the target speaker. Earlier F5 similarity measurements did not establish the perceived likeness the user wanted. These experiments make that comparison accessible and investigate extra spoken words before making another quality claim.

## Fast delivery controller and acoustic scope

The new controller is bounded deterministic code, not a new language model. It keeps a fitting passage intact, including adjacent sentences and paragraph whitespace. When an engine limit requires a split, it prefers paragraphs, sentence endings, clauses and then complete words. F5 planning accounts for its reference-plus-output context, UTF-8 duration heuristic, short-text slowdown and 50 ms conditioning tail. SOAR/MF2 use their 150-character low-memory limit. Source offsets and word order are preserved; no emotion instructions are inserted into the spoken text.

The F5 fallback join no longer overlaps 150 ms of independently generated speech. It retains all generated frames and adds only missing punctuation silence after accounting for existing quiet at both edges. This removes a source of consonant blending, but does not prove that the result sounds natural.

CPU reference analysis is cached by recording hash, transcript hash, VAD pin, policy and runtime version. On this laptop the measured first analysis took 2.675 s and a warm cache hit 6.63 ms. Short-passage planning on the local workstation took a median approximately 0.006 ms; a 2,849-character input took 37.4 ms across 25 runs. A live GPU request measured 0.008 ms planning. These are controller timings, separate from model loading and neural synthesis.

A small text language model could select delivery instructions, but the selected synthesizer would still need to support them. A text-only planner cannot hear the user's accent or manufacture the reference's emotion. CosyVoice 3 has a native instruction route; its frontend removes the LLM reference speech-token prompt in that mode, so instructions and automatic reference-style transfer need separate evaluation. Its stock checkpoint/runtime footprint also needs a new memory/disk feasibility check on this 4 GB laptop. No unsupported emotion instructions or additional large model downloads were added here. [Official CosyVoice example](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/example.py), [instruction frontend](https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/cli/frontend.py).

| Requested aspect | Current handling | Limit |
|---|---|---|
| Pitch / frequency | Approximate reference F0 median and range; learned acoustic conditioning | No automatic pitch-shift patch or exact contour promise |
| Volume | RMS, speech RMS, peaks and output headroom | Recording gain is not a measure of lung pressure or emotion |
| Timbre / tone | Complete acoustic prompt and matching transcript | Similarity scores cannot establish perceived identity |
| Pace | Cached whole-clip and speech-only rates; F5 model pace multiplier | SOAR/MF2 preserve native delivery at 1.0× |
| Pauses | Unpadded VAD gaps; paragraph/punctuation context; conservative fallback joins | No word-level pause alignment is inferred |
| Melody / inflection | Preserve connected sentence context for the learned model | No exact rise/fall transfer is guaranteed |
| Accent / dialect | Reference-conditioned synthesis and manual engine comparison | No accent label is guessed from pitch or spectrum |
| Vocal habits | Reference examples plus descriptive cadence measurements | One short utterance does not reveal habitual range |
| Emotion / expression | A reference demonstrating the desired delivery can condition the model | No inferred emotion label or guaranteed emotion control |

These distinctions follow the broader separation between frequency/intensity measurements and voice quality or learned delivery. See [NCVS Voice Qualities](https://ncvs.org/tutorials/voice-qualities/) and the [official Dots continuation-cloning documentation](https://github.com/studio-dots-ai/dots.tts/blob/main/README.md). The current 2.7-second recording contains roughly 1.45 seconds of detected speech. The studio now displays nonblocking guidance to use more clean speech in the intended accent/expression and to review every transcript word.

## Same reference, same sentences

The latest own-voice profile, `e56d7a105b65766d41634460a8b1c59bc46bd06a23c2170e13226c4484aa3d76`, contains a complete **2.6987-second, 48 kHz** recording. Its original WAV SHA-256 is `c0066076962dec6a42345213548d0d9220f9f7d9fdbe61ba259c433897380680`; the recording was retained throughout.

Two natural sentences, with **27 expected words in total**, were generated at seed 42:

- “I was going to call you earlier, but the day just got away from me.”
- “Wait, are you serious? I thought we were meeting tomorrow, not today.”

F5, SOAR and MF2 ran sequentially, with an excluded warmup and closed workers between models. The existing T-1000 installation at `/home/prieyan/weeb/T-1000` supplied its environment and models without new installs or model copies. It uses dots-tts runtime `b65ca7149fb43b8fb8d52916aa12738deda7b188`, Torch 2.8.0+cu128, NF4 language-model blocks, BF16 acoustic core and FP32 speaker/vocoder modules. SOAR uses ten Euler steps and CFG 1.2; MF2 retains its trained two-step/CFG 0/tau 1.3 configuration. Both preserve complete reference audio and its transcript context.

## Initial results and the prompt-word problem

The initial saved and freshly recognized reference transcript was `Hello, this is Dijon`, without terminal punctuation. This was an ASR hypothesis, not a reviewed spelling of the user's name.

| Route, two requests each | Valid outputs | ASR word errors / 27 | Median generation | Median adapter first host PCM | Median RTF | Peak CUDA reserved | Median independent cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| F5, 32 steps | 2/2 | 2/27 | 3.7175 s | 3.7403 s | 0.4787 | 786 MiB | 0.55366 |
| T-1000 SOAR, raw reference text | 2/2 | 10/27 | 22.9771 s | 23.0025 s | 2.8960 | 3556 MiB | 0.67513 |
| T-1000 MF2, raw reference text | 2/2 | 5/27 | 6.5180 s | 6.5380 s | 1.0438 | 3548 MiB | 0.75527 |

All six WAVs were finite, nonempty, mono and unclipped at export. All four T-1000 requests reached observed EOS rather than a generation cap. T-1000 output remains native 48 kHz; F5 output remains 24 kHz. Output rate does not establish perceived quality.

The recognizer found prompt-like words in generated audio: the raw MF2 statement began `Hello, this is Dishon` before the requested sentence; the SOAR question began `This is Dijon Wade`; the F5 statement included `I joined` before its requested words. These findings required inspection even though the independent cosine was higher for T-1000. Repeating reference words can increase apparent similarity while harming the requested speech. **Cosine is not a likeness percentage or a winner criterion.**

Peak handling is explicit. For example, F5's first unscaled peak was 1.487 and its recorded multiplicative gain was 0.659. Output samples were attenuated to leave a 0.98 peak margin. The reports retain raw peaks, raw full-scale fractions and applied gain; unclipped exported PCM does not establish that raw model samples stayed below full scale. No post-hoc pitch or tempo correction was applied.

## Boundary investigation and controlled checks

The original T-1000 application strips outer transcript whitespace but does not append sentence punctuation. The pinned runtime adds a newline to reference text, then combines it with new text inside a text/audio generation template. The new worker uses that same contract; there was no accidental concatenation of adjacent words. Prompt acoustic latents are prefilled, and the model discards a regenerated prompt-tail patch. This is not forced word alignment. [Pinned runtime](https://github.com/studio-dots-ai/dots.tts/blob/b65ca7149fb43b8fb8d52916aa12738deda7b188/src/dots_tts/runtime.py).

Silero VAD identified speech at **0.252–1.720 seconds**, about **1.468 seconds** inside the 2.699-second clip. An optional 150 ms padded edge branch would span 0.102–1.870 seconds. The excluded trailing region had RMS 0.0597 versus detected-speech RMS 0.2619; it was not dead silence. The training pipeline contains a separate edge-normalization operation with 250 ms edges and a relative 30 dB threshold, while the T-1000 application intentionally preserves complete prompts. These differences suggest a testable conditioning issue, not proof that trimming is universally better. [Pinned edge preprocessing](https://github.com/studio-dots-ai/dots.tts/blob/b65ca7149fb43b8fb8d52916aa12738deda7b188/src/dots_tts/data/pipelines/preprocessing.py).

The first controlled candidate kept the original full audio and added only a period: `Hello, this is Dijon.` Both MF2 outputs then matched their expected ASR text exactly: **0/27 word errors and zero character errors**, with observed EOS and valid 48 kHz exports. Median generation was **4.8870 seconds**, adapter first host PCM **4.9063 seconds**, RTF **1.1109**, reserved memory **3540 MiB**, and independent cosine **0.75626**. The old spelling was retained for this punctuation-only check; no name was guessed.

That diagnostic driver initially had a list-alias bug: appending candidate records extended the control loop. The exact-pair guard rejected the duplicate after two valid period-only outputs. Those outputs and the last valid checkpoint were retained and rescored on CPU in a new directory. The alias was fixed and an end-to-end fake-backend regression now checks the exact number of generated requests. The failure was in the harness, not model inference.

The user then explicitly confirmed the words **“hello this is dijo.”** The profile transcript was updated to `Hello, this is Dijo.` while retaining the original audio. A fresh MF2 trial used that reviewed text, the same original full recording, the same two requested sentences and seed 42. Both outputs again had **0/27 ASR word errors, zero character errors, true EOS and finite, unclipped 48 kHz PCM**.

| Reviewed Dijo trial | Result |
|---|---:|
| Eligible requests | 2/2 |
| Median generation time | 4.8782 s |
| Median adapter first host PCM | 4.9007 s |
| Median RTF | 1.1089 |
| Maximum CUDA reserved memory | 3540 MiB |
| Median independent speaker cosine | 0.76315 |

The raw baseline and reviewed trial differ in both spelling and punctuation. The intervening punctuation-only check isolates that change and already removed the observed insertions on these sentences. Two sentences from one short reference cannot establish general transcription accuracy, emotional delivery or identity preservation.

## Implemented reference-text policy

The isolated T-1000 adapter now defaults to `reference_boundary="sentence"`. It adds a neutral terminal period only when the conditioning transcript lacks `.`, `!`, `?` or `…`, accounting for closing quotation marks. Existing terminal punctuation and all spoken words are retained. It does not choose a name, infer an accent or assign an emotion.

The source profile transcript and original recording remain unchanged by this policy. The cache records `source_transcript`, `conditioning_text`, the boundary mode and policy version. The conditioning signature includes the policy; request provenance records it too. `reference_boundary="raw"` remains available for reproducing the original control. The comparison CLI accepts `--reference-boundary raw`; its normal sentence-boundary route has a distinct label.

**Reference VAD cropping is not a default.** The reviewed full-reference trial passed this small word-accuracy screen, so further cropping was not needed for this check. The derived edge candidate and its measurements remain documented; no speech or pauses were removed from the user's source recording. A cheap deterministic punctuation rule is the measured boundary fix here, not a new trained decision model or a claim that emotion matching is solved.

## Artifacts and limits

| Run | Report | Reference-guided listening |
|---|---|---|
| Original six F5/SOAR/MF2 outputs | [report.json](../data/benchmarks/delivery-own-first/report.json) | [LISTEN.html](../data/benchmarks/delivery-own-first/LISTEN.html) |
| Old-name period-only check, CPU rescore | [report.json](../data/benchmarks/delivery-period-scored/report.json) | [LISTEN.html](../data/benchmarks/delivery-period-scored/LISTEN.html) |
| User-reviewed Dijo, full reference | [report.json](../data/benchmarks/delivery-reviewed-Dijo-first/report.json) | [LISTEN.html](../data/benchmarks/delivery-reviewed-Dijo-first/LISTEN.html) |

The original incomplete diagnostic directory, `data/benchmarks/delivery-boundaries-first/`, is also retained. Runtime artifacts are private local recordings and are ignored by Git. Matching run directories remain on the GPU laptop under `/home/prieyan/T-2000/data/benchmarks/`.

Generation times exclude model loading and later CPU scoring. Adapter first host PCM includes worker request/preparation/transfer and waits for the complete phrase, so it is not browser audible latency. T-1000 reserved memory excludes driver and desktop allocations; only a small GPU margin remained. Existing host RAM/swap pressure made some cold model loads take over a minute. The independent WeSpeaker model and Nemotron ASR provide diagnostics, not human naturalness or emotion ratings.

The integration passed **192 Python tests**, Ruff and AudioWorklet checks locally. The adapter and boundary tooling account for 34 focused tests. Live Chromium verification changed the running engine from F5 to MF2, preserved the selected recording and reviewed transcript, locked native delivery at 1.0×, prepared the reference, generated a connected two-sentence utterance in one context, played it and downloaded the WAV. Output was mono 48 kHz; generation took 5.715 seconds and planning 0.008 ms. No page errors or mobile horizontal overflow occurred. These are functional observations, not an accent or emotion rating.

MF2 is the selected laptop starting configuration for further listening. SOAR, F5 and Melo/OpenVoice remain available through Voice engine; changing engines releases the current worker before another loads. The service reuses `/home/prieyan/weeb/T-1000/.venv/bin/python` and its existing models, without another model download. The portable source default remains Melo/OpenVoice.

Live artifacts: [browser verification](../data/validation/reference-delivery/mf2-live-browser.json), [desktop view](../data/validation/reference-delivery/mf2-desktop.png), [mobile view](../data/validation/reference-delivery/mf2-mobile.png), [native MF2 sample](../data/validation/reference-delivery/mf2-live.wav), and [planner timing](../data/validation/reference-delivery/planner-benchmark.json).

Accent, flow, naturalness, pauses, emotion and perceived speaker likeness must still be assessed by listening against the user's own reference; no route has been declared the perceptual winner.

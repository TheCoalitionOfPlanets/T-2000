const $ = (id) => document.getElementById(id);
const MAX_OUTPUT_SECONDS = 120;
const ENGINE_LABELS = {
  't1000-soar': 'T-1000 SOAR · Reference quality',
  't1000-mf2': 'T-1000 MF2 · Faster reference',
  'f5-tts-v1': 'F5 · Direct voice',
  'melo-openvoice-v2': 'Melo/OpenVoice · Fast',
};
const state = {
  profiles: [], selected: null, draft: null, draftUrl: null, previewUrl: null, previewDuration: null, previewValid: false,
  capture: null, capturing: false, saving: false,
  socket: null, generating: false, playback: null, run: null, sttAvailable: null,
  maxReferenceSeconds: 30, recommendedReferenceSeconds: '10–20', runtimeReady: false,
  currentBackend: null, availableBackends: [],
  speedControlEnabled: true, lastAdjustableSpeed: '1',
  referenceDeliveryProfile: null,
};

function status(message, error = false) {
  $('session-status').textContent = message;
  $('session-status').classList.toggle('error', error);
}

function explain(error) {
  return error?.message || String(error);
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((item) => item.msg).join(' ');
      else message = body.message || body.error || message;
    } catch { /* The HTTP status remains actionable if the server has no JSON body. */ }
    throw new Error(message);
  }
  return response.json();
}

function rememberProfile(id) {
  try { localStorage.setItem('t2000.profile', id); } catch { /* Storage can be disabled. */ }
}

function lastProfile() {
  try { return localStorage.getItem('t2000.profile'); } catch { return null; }
}

function updateActions() {
  const selectedTooLong = state.selected?.duration_seconds > state.maxReferenceSeconds;
  const previewTooLong = state.previewDuration > state.maxReferenceSeconds;
  const previewTooShort = !Number.isFinite(state.previewDuration) || state.previewDuration < 1;
  $('generate-button').disabled = state.generating || state.capturing || state.saving || Boolean(state.draft) || !state.selected || selectedTooLong || !textValue().trim();
  $('stop-button').disabled = !(state.generating || (state.playback && !state.playback.drained && !state.playback.cancelled));
  $('save-reference').disabled = !state.draft || !state.previewValid || previewTooLong || previewTooShort || state.capturing || state.saving || state.generating;
  $('save-transcript').disabled = state.generating || state.saving;
  $('prepare-voice').disabled = state.generating || state.saving || state.capturing || selectedTooLong;
  $('profile-select').disabled = state.capturing || state.saving || state.generating;
  $('reference-file').disabled = !state.runtimeReady || state.capturing || state.saving || state.generating;
  $('record-button').disabled = !state.runtimeReady || state.saving || state.generating;
  $('reference-editor').disabled = state.saving || state.generating || state.capturing;
  $('edit-reference').disabled = state.saving || state.generating || state.capturing;
  $('voice-engine').disabled = !state.runtimeReady || state.saving || state.generating || state.capturing || state.availableBackends.length < 2;
  for (const id of ['speech-text', 'speech-speed', 'speech-seed', 'sentence-gap', 'reference-transcript', 'voice-name', 'auto-transcribe']) $(id).disabled = state.saving;
  $('speech-speed').disabled = state.saving || !state.speedControlEnabled;
  if (!state.speedControlEnabled) { $('speech-speed').value = '1'; renderSpeed(); }
  $('empty-output').hidden = state.generating || !$('output-section').hidden;
  $('listening-room').setAttribute('aria-busy', String(state.generating));
  $('stop-button').hidden = $('stop-button').disabled;
  for (const button of document.querySelectorAll('[data-example]')) button.disabled = state.generating || state.saving;
}

function textValue() { return $('speech-text').value; }

function renderSpeed() {
  $('speed-value').value = `${Number($('speech-speed').value).toFixed(2).replace(/0$/, '')}×`;
}

function renderSentenceGap() {
  $('sentence-gap-value').value = `${Number($('sentence-gap').value).toFixed(2)} s`;
}

function applyRuntime(runtime) {
    const ready = Boolean(runtime.loaded);
    const directVoice = runtime.backend === 'f5-tts-v1';
    const t1000 = runtime.backend === 't1000-soar' || runtime.backend === 't1000-mf2';
    const referenceGuided = directVoice || t1000;
    const speedControlEnabled = typeof runtime.speed_control?.enabled === 'boolean' ? runtime.speed_control.enabled : !t1000;
    if (state.speedControlEnabled && !speedControlEnabled) state.lastAdjustableSpeed = $('speech-speed').value;
    if (!state.speedControlEnabled && speedControlEnabled) $('speech-speed').value = state.lastAdjustableSpeed;
    state.speedControlEnabled = speedControlEnabled;
    if (!speedControlEnabled) $('speech-speed').value = '1';
    renderSpeed();
    state.currentBackend = runtime.backend;
    const supported = Array.isArray(runtime.available_backends)
      ? runtime.available_backends.filter((item) => item?.available === true && Object.hasOwn(ENGINE_LABELS, item.id))
      : [];
    state.availableBackends = supported;
    $('voice-engine').replaceChildren();
    for (const engine of supported) $('voice-engine').append(new Option(ENGINE_LABELS[engine.id], engine.id));
    if (!supported.length && Object.hasOwn(ENGINE_LABELS, runtime.backend)) $('voice-engine').append(new Option(ENGINE_LABELS[runtime.backend], runtime.backend));
    $('voice-engine').value = runtime.backend;
    $('engine-field').hidden = supported.length < 2;
    state.runtimeReady = true;
    state.maxReferenceSeconds = Number.isFinite(runtime.max_reference_seconds) && runtime.max_reference_seconds >= 1 && runtime.max_reference_seconds <= 30
      ? runtime.max_reference_seconds : t1000 ? 6 : directVoice ? 12 : 30;
    state.recommendedReferenceSeconds = t1000
      ? `3–${state.maxReferenceSeconds}`
      : directVoice ? '8–12' : '10–20';
    $('reference-description').textContent = `For voice likeness, start with ${state.recommendedReferenceSeconds} seconds of clean speech. One voice, with as little background noise as possible.`;
    $('help-reference-guidance').textContent = `Record or upload ${state.recommendedReferenceSeconds} seconds of clear, single-speaker speech, up to ${state.maxReferenceSeconds} seconds. Listen back for noise, clipping, and long silences. Shorter clips still work.`;
    if (!state.capturing && !state.draft) $('capture-status').textContent = `Aim for ${state.recommendedReferenceSeconds} seconds of clear speech. Up to ${state.maxReferenceSeconds} seconds.`;
    referenceGuidance(draftReferenceDuration());
    $('engine-name').textContent = ENGINE_LABELS[runtime.backend] || 'Local voice engine';
    $('speech-speed-label').textContent = !speedControlEnabled ? 'Reference delivery' : referenceGuided ? 'Reference pace' : 'Speaking speed';
    $('speech-speed-help').textContent = !speedControlEnabled
      ? 'Uses the recording’s delivery. Speed adjustment is available with F5 or Melo.' : directVoice
      ? '1.0× follows the reference’s estimated pace; adjust for new text.'
      : 'Adjust the speaking speed for your new text.';
    $('backend-guidance').textContent = t1000
      ? 'The recording and its exact transcript guide the generated voice and delivery. Accent, sentence flow and emotion can still differ; listen beside your reference before choosing an engine.' : directVoice
      ? 'The recording and its transcript guide the voice, rhythm and estimated pace. At 1.0×, the model follows that pace. Pitch, tone, smoothness and expression can still differ; compare the result with your recording.'
      : 'Your recording guides the voice color. Use Speaking speed for pace, and listen for pitch and delivery differences.';
    const planner = runtime.delivery_planner;
    $('delivery-planner-status').hidden = !(planner === true || (planner && typeof planner === 'object' && planner.enabled === true));
    $('runtime-status').classList.toggle('ready', ready);
    $('runtime-status').replaceChildren(Object.assign(document.createElement('i'), { ariaHidden: 'true' }), document.createTextNode(ready ? 'Voice engine ready' : 'Voice engine loads on demand'));
    if (typeof runtime.stt?.available === 'boolean') {
      state.sttAvailable = runtime.stt.available;
      transcriptStatus();
    }
    updateActions();
}

async function refreshStatus() {
  try {
    applyRuntime(await api('/api/status'));
  } catch {
    $('runtime-status').classList.remove('ready');
    $('runtime-status').textContent = 'Studio disconnected';
  }
}

async function switchBackend() {
  const requested = $('voice-engine').value;
  if (requested === state.currentBackend || state.saving || state.generating || state.capturing) return;
  if (!state.availableBackends.some((item) => item.id === requested)) return;
  const previous = state.currentBackend;
  const selectedId = state.selected?.id;
  state.saving = true;
  updateActions();
  status('Changing voice engine. Your recording and saved voices stay here…');
  try {
    if (state.selected && !state.draft) {
      const transcript = $('reference-transcript').value.trim();
      if (transcript !== (state.selected.transcript || '').trim()) {
        const profile = await api(`/api/profiles/${encodeURIComponent(selectedId)}`, {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ transcript }),
        });
        state.selected = profile;
        const index = state.profiles.findIndex((item) => item.id === profile.id);
        if (index >= 0) state.profiles[index] = profile;
      }
    }
    const runtime = await api('/api/backend', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ backend: requested }),
    });
    applyRuntime(runtime);
    if (!state.draft) await refreshProfiles(selectedId);
    const tooLong = draftReferenceDuration() > state.maxReferenceSeconds;
    status(tooLong ? `Engine changed. Choose a reference of ${state.maxReferenceSeconds} seconds or less for this engine.`
      : state.draft ? state.previewValid ? 'Engine changed. Save this previewed reference to continue.' : 'Engine changed. Preview your edits, then save this reference.'
      : 'Engine changed. Prepare your voice, then compare it with your recording.', tooLong);
  } catch (error) {
    $('voice-engine').value = previous;
    await refreshStatus();
    status(explain(error), true);
  } finally {
    state.saving = false;
    updateActions();
  }
}

function transcriptStatus(message, error = false) {
  $('transcript-status').textContent = message || (state.sttAvailable === false
    ? 'Speech recognition is unavailable. You can type the exact words from your recording.'
    : state.selected && !state.draft
      ? 'Review these words against the recording. You can edit and save corrections.'
      : 'New recordings are transcribed when saved. Any transcript you type is kept.');
  $('transcript-status').classList.toggle('error', error);
}

function draftReferenceDuration() {
  return state.draft ? state.previewValid ? state.previewDuration : state.draft.duration : state.selected?.duration_seconds;
}

function referenceGuidance(duration) {
  const tooShort = Number.isFinite(duration) && duration < 1;
  const tooLong = Number.isFinite(duration) && duration > state.maxReferenceSeconds;
  const short = Number.isFinite(duration) && duration >= 1 && duration < 5;
  const delivery = state.referenceDeliveryProfile && !state.draft && state.referenceDeliveryProfile.profileId === state.selected?.id
    ? state.referenceDeliveryProfile.data : null;
  const speechSeconds = delivery?.speech_seconds;
  const sparseSpeech = Number.isFinite(speechSeconds) && speechSeconds >= 0
    && (speechSeconds < 2 || (Number.isFinite(delivery?.speech_fraction) && delivery.speech_fraction < 0.6));
  $('reference-guidance').hidden = !(tooShort || tooLong || sparseSpeech || short);
  $('reference-guidance').textContent = tooShort ? 'Record or upload at least one second of clear speech.' : tooLong
    ? `This reference is ${duration.toFixed(1)} seconds. This voice engine supports references up to ${state.maxReferenceSeconds} seconds. ${state.draft ? 'Trim it and preview the kept speech before saving.' : 'Use Clean up or trim this voice to save a shorter copy.'}`
    : sparseSpeech ? `Only about ${speechSeconds.toFixed(1)} seconds of speech were detected. Use a longer, clean recording in the accent and expression you want, and check every transcript word.`
    : short ? `This short reference can be used. For more voice detail, try ${state.recommendedReferenceSeconds} seconds of clean speech.` : '';
}

async function refreshProfiles(preferred) {
  const profiles = await api('/api/profiles');
  state.profiles = Array.isArray(profiles) ? profiles : profiles.profiles;
  $('profile-select').replaceChildren();
  if (!state.profiles.length) {
    const option = new Option('No voices yet', '');
    $('profile-select').append(option);
    state.selected = null;
    $('selected-voice').classList.remove('is-ready');
    updateActions();
    return;
  }
  for (const profile of state.profiles) $('profile-select').append(new Option(profile.name, profile.id));
  const id = preferred || state.selected?.id || lastProfile();
  const selected = state.profiles.find((profile) => profile.id === id) || state.profiles[0];
  $('profile-select').value = selected.id;
  selectProfile(selected.id);
}

function clearDraft() {
  if (state.draftUrl) URL.revokeObjectURL(state.draftUrl);
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  state.previewValid = false;
  state.previewDuration = null;
  state.draftUrl = null;
  state.draft = null;
  $('reference-file').value = '';
  $('draft-fields').hidden = true;
  $('transcribe-label').hidden = true;
  $('save-reference').hidden = true;
  $('compare-reference').hidden = true;
}

function selectProfile(id) {
  const profile = state.profiles.find((item) => item.id === id);
  if (!profile) return;
  if (state.selected?.id !== id) state.referenceDeliveryProfile = null;
  clearDraft();
  state.selected = profile;
  $('selected-voice').classList.add('is-ready');
  rememberProfile(id);
  $('reference-transcript').value = profile.transcript || '';
  transcriptStatus();
  $('save-transcript').hidden = false;
  $('prepare-voice').hidden = false;
  $('edit-reference').hidden = false;
  $('reference-player').src = `/api/profiles/${encodeURIComponent(id)}/audio`;
  $('reference-player').hidden = false;
  $('capture-status').textContent = `Record or upload a new sample, up to ${state.maxReferenceSeconds} seconds, to add another voice.`;
  referenceGuidance(profile.duration_seconds);
  $('reference-summary').hidden = false;
  $('reference-summary').textContent = `${profile.duration_seconds.toFixed(1)} seconds · ${(profile.sample_rate / 1000).toFixed(1)} kHz · reference cached`;
  $('analysis-panel').hidden = false;
  drawAnalysis(profile.analysis || {});
  status(profile.duration_seconds > state.maxReferenceSeconds ? `Choose a reference of ${state.maxReferenceSeconds} seconds or less for this voice engine.` : 'Ready when you are.', profile.duration_seconds > state.maxReferenceSeconds);
  updateActions();
}

function setDraft(blob, filename, sampleRate, duration) {
  clearDraft();
  state.referenceDeliveryProfile = null;
  state.draft = { blob, filename, duration };
  $('selected-voice').classList.remove('is-ready');
  state.draftUrl = URL.createObjectURL(blob);
  $('reference-player').src = state.draftUrl;
  $('reference-player').hidden = false;
  $('draft-fields').hidden = false;
  $('edit-reference').hidden = true;
  $('trim-start').value = '0';
  $('trim-end').value = '';
  $('noise-reduction').checked = true;
  $('auto-trim').checked = false;
  $('cleanup-status').textContent = 'Preview your changes before saving. You are listening to the original.';
  $('voice-name').value = filename.replace(/\.wav$/i, '') === 'recording' ? 'My voice' : filename.replace(/\.wav$/i, '').slice(0, 80);
  $('reference-transcript').value = '';
  transcriptStatus();
  $('transcribe-label').hidden = false;
  $('save-reference').hidden = false;
  $('save-transcript').hidden = true;
  $('prepare-voice').hidden = true;
  $('reference-summary').hidden = true;
  $('analysis-panel').hidden = true;
  $('capture-status').textContent = sampleRate ? `${duration.toFixed(1)} seconds captured at ${(sampleRate / 1000).toFixed(1)} kHz. Listen, then save your reference.` : `${filename} · Listen, then save your reference.`;
  referenceGuidance(duration);
  status('Preview your cleanup and trim settings, then save this reference to use it for generation.');
  updateActions();
}

function processingData() {
  const data = new FormData();
  data.append('file', state.draft.blob, state.draft.filename);
  const start = Number($('trim-start').value);
  const end = $('trim-end').value.trim() === '' ? null : Number($('trim-end').value);
  if ($('trim-start').value.trim() === '' || !Number.isFinite(start) || start < 0 ||
      (end !== null && (!Number.isFinite(end) || end <= start))) {
    throw new Error('Choose a valid start and end time. Keep at least one second of speech.');
  }
  data.append('start_seconds', String(start));
  if (end !== null) data.append('end_seconds', String(end));
  data.append('noise_reduction', String($('noise-reduction').checked));
  data.append('auto_trim', String($('auto-trim').checked));
  return data;
}

function invalidatePreview(clearTranscript = false) {
  if (!state.draft) return;
  state.previewValid = false;
  state.previewDuration = null;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  $('compare-reference').hidden = true;
  $('reference-player').src = state.draftUrl;
  $('cleanup-status').textContent = 'Changes need a new preview before saving. You are listening to the original.';
  if (clearTranscript) $('reference-transcript').value = '';
  referenceGuidance(state.draft.duration);
  updateActions();
}

async function previewReference() {
  if (!state.draft || state.saving || state.capturing || state.generating) return;
  state.saving = true; updateActions();
  $('cleanup-status').textContent = 'Preparing your preview…';
  try {
    const response = await fetch('/api/reference/preview', { method: 'POST', body: processingData() });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === 'string' ? body.detail : 'The reference could not be processed. Check your crop times.');
    }
    const duration = Number(response.headers.get('X-Reference-Duration'));
    if (!Number.isFinite(duration) || duration < 1) throw new Error('The preview did not contain at least one second of audio.');
    const blob = await response.blob();
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(blob);
    state.previewValid = true;
    state.previewDuration = duration;
    $('reference-player').src = state.previewUrl;
    $('compare-reference').hidden = false;
    $('compare-reference').textContent = 'Listen to original';
    referenceGuidance(duration);
    $('cleanup-status').textContent = `Preview ready${Number.isFinite(duration) && duration > 0 ? ` · ${duration.toFixed(2)} seconds` : ''}. Listen and check that the transcript matches the kept words, then save.`;
    status(duration > state.maxReferenceSeconds ? `Preview is ${duration.toFixed(2)} seconds. Trim it to ${state.maxReferenceSeconds} seconds or less for this engine, then preview again.` : 'Preview ready. Save this reference when it sounds right.', duration > state.maxReferenceSeconds);
  } catch (error) {
    invalidatePreview();
    $('cleanup-status').textContent = explain(error);
    status(explain(error), true);
  } finally { state.saving = false; updateActions(); }
}

async function editReference() {
  if (!state.selected || state.saving || state.generating || state.capturing) return;
  const selected = state.selected;
  state.saving = true; updateActions();
  try {
    const response = await fetch(`/api/profiles/${encodeURIComponent(selected.id)}/audio`);
    if (!response.ok) throw new Error('The saved recording is unavailable. Upload or record it again.');
    setDraft(await response.blob(), `${selected.name}.wav`, selected.sample_rate, selected.duration_seconds);
    $('voice-name').value = `${selected.name} edited`.slice(0, 80);
    status('Edit this copy, then preview and save. Your saved voice remains available.');
  } catch (error) { status(explain(error), true); }
  finally { state.saving = false; updateActions(); }
}

function floatWave(chunks, frames, sampleRate) {
  const buffer = new ArrayBuffer(44 + frames * 4);
  const view = new DataView(buffer);
  const word = (offset, value) => [...value].forEach((letter, index) => view.setUint8(offset + index, letter.charCodeAt(0)));
  word(0, 'RIFF'); view.setUint32(4, 36 + frames * 4, true); word(8, 'WAVE');
  word(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 3, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true); view.setUint16(32, 4, true); view.setUint16(34, 32, true);
  word(36, 'data'); view.setUint32(40, frames * 4, true);
  let offset = 44;
  for (const chunk of chunks) for (const sample of chunk) { view.setFloat32(offset, sample, true); offset += 4; }
  return new Blob([buffer], { type: 'audio/wav' });
}

async function startCapture() {
  if (state.capturing || state.generating) return;
  const maxSeconds = state.maxReferenceSeconds;
  state.capturing = true;
  updateActions();
  $('record-label').textContent = 'Starting…';
  let context, stream;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) throw new Error('Microphone recording requires a browser with AudioWorklet on localhost or HTTPS.');
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 } });
    context = new AudioContext(); // Preserve the device/context native sample rate.
    await context.audioWorklet.addModule('/static/playback-worklet.js');
    await context.resume();
    const node = new AudioWorkletNode(context, 'reference-capture', { processorOptions: { maxFrames: Math.floor(context.sampleRate * maxSeconds) }, numberOfOutputs: 1, outputChannelCount: [1] });
    const source = context.createMediaStreamSource(stream);
    const mute = context.createGain(); mute.gain.value = 0;
    source.connect(node); node.connect(mute); mute.connect(context.destination);
    const capture = { context, stream, node, source, mute, chunks: [], frames: 0, stopping: false, timer: null, watchdog: null };
    state.capture = capture;
    node.onprocessorerror = () => {
      if (state.capture === capture) failCapture(capture, 'The microphone processor stopped. Please record again.');
    };
    node.port.onmessage = ({ data }) => {
      if (state.capture !== capture) return;
      if (data.type === 'samples') {
        if (capture.frames + data.samples.length > Math.floor(context.sampleRate * maxSeconds)) {
          failCapture(capture, 'The recording exceeded its limit. Please record again.');
          return;
        }
        capture.chunks.push(data.samples);
        capture.frames += data.samples.length;
      }
      if (data.type === 'stopped') finishCapture(capture, data);
    };
    capture.started = performance.now();
    capture.timer = setInterval(() => {
      const elapsed = Math.min((performance.now() - capture.started) / 1000, maxSeconds);
      $('capture-status').textContent = `Recording ${elapsed.toFixed(1)} / ${maxSeconds} seconds. Click Stop recording when finished.`;
      if (elapsed >= maxSeconds) stopCapture();
    }, 100);
    $('record-button').classList.add('recording');
    $('record-label').textContent = 'Stop recording';
    status('Recording your reference voice.');
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop());
    if (context) await context.close();
    state.capturing = false;
    $('record-label').textContent = 'Record voice';
    updateActions();
    status(explain(error), true);
  }
}

function releaseCapture(capture) {
  clearInterval(capture.timer); clearTimeout(capture.watchdog);
  capture.node.disconnect(); capture.source.disconnect(); capture.mute.disconnect();
  capture.stream.getTracks().forEach((track) => track.stop());
  void capture.context.close();
  state.capture = null;
  state.capturing = false;
  $('record-button').classList.remove('recording');
  $('record-label').textContent = 'Record voice';
  updateActions();
}

function failCapture(capture, message) {
  releaseCapture(capture);
  status(message, true);
}

function finishCapture(capture, data) {
  // The stopped acknowledgement follows the worklet's final partial batch.
  if (data.frames !== capture.frames || data.sample_rate !== capture.context.sampleRate) {
    failCapture(capture, 'The final recording samples were incomplete. Please record again.');
    return;
  }
  const rate = capture.context.sampleRate;
  const duration = capture.frames / rate;
  const blob = floatWave(capture.chunks, capture.frames, rate);
  releaseCapture(capture);
  if (duration < 1) { status('Record at least one second of speech.', true); return; }
  setDraft(blob, 'recording.wav', rate, duration);
}

function stopCapture() {
  const capture = state.capture;
  if (!capture || capture.stopping) return;
  capture.stopping = true;
  clearInterval(capture.timer);
  $('record-label').textContent = 'Finishing…';
  capture.node.port.postMessage({ type: 'stop' });
  capture.watchdog = setTimeout(() => {
    if (state.capture === capture) failCapture(capture, 'The microphone did not finalize. Please record again.');
  }, 2000);
}

async function saveReference() {
  if (!state.draft || !state.previewValid || state.saving || state.capturing || state.generating) return;
  if (!Number.isFinite(state.previewDuration) || state.previewDuration < 1) { status('Preview at least one second of speech before saving.', true); return; }
  if (state.previewDuration > state.maxReferenceSeconds) { status(`Trim the reference to ${state.maxReferenceSeconds} seconds or less, then preview again.`, true); return; }
  state.saving = true; updateActions();
  const transcribing = $('auto-transcribe').checked && !$('reference-transcript').value.trim();
  status(transcribing ? 'Saving and transcribing the reference…' : 'Analyzing and saving the reference…');
  transcriptStatus(transcribing ? 'Transcribing your recording. Review the words when it is saved…' : 'Saving your reference and transcript…');
  $('save-reference').textContent = transcribing ? 'Saving and transcribing…' : 'Saving reference…';
  try {
    const data = processingData();
    data.append('name', $('voice-name').value.trim() || 'My voice');
    data.append('transcript', $('reference-transcript').value.trim());
    data.append('transcribe', String($('auto-transcribe').checked));
    const profile = await api('/api/profiles', { method: 'POST', body: data });
    await refreshProfiles(profile.id);
    if (profile.transcription_error) transcriptStatus(`Transcription unavailable: ${profile.transcription_error} You can type the words and save them.`, true);
    else transcriptStatus(profile.transcript ? 'Transcript saved. Listen to the recording and correct any missed words.' : 'No transcript was returned. You can type the words from the recording.');
    status(profile.transcription_error ? `Reference saved. Transcription unavailable: ${profile.transcription_error}` : 'Reference saved. Your voice is ready to use.', Boolean(profile.transcription_error));
  } catch (error) { status(explain(error), true); }
  finally { state.saving = false; $('save-reference').textContent = 'Save reference voice'; updateActions(); }
}

async function saveTranscript() {
  if (!state.selected || state.saving) return;
  state.saving = true; updateActions();
  try {
    const profile = await api(`/api/profiles/${encodeURIComponent(state.selected.id)}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ transcript: $('reference-transcript').value.trim() }),
    });
    const index = state.profiles.findIndex((item) => item.id === profile.id);
    if (index >= 0) state.profiles[index] = profile;
    state.selected = profile;
    status('Transcript saved.');
  } catch (error) { status(explain(error), true); }
  finally { state.saving = false; updateActions(); }
}

async function prepareVoice() {
  if (!state.selected || state.generating || state.saving) return;
  if (state.selected.duration_seconds > state.maxReferenceSeconds) { status(`Choose a reference of ${state.maxReferenceSeconds} seconds or less before preparing this voice.`, true); return; }
  state.saving = true; updateActions();
  status('Preparing this voice. The first model load can take a while…');
  try {
    const profileId = state.selected.id;
    const transcript = $('reference-transcript').value.trim();
    if (transcript !== (state.selected.transcript || '').trim()) {
      await api(`/api/profiles/${encodeURIComponent(profileId)}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ transcript }),
      });
    }
    const prepared = await api(`/api/profiles/${encodeURIComponent(profileId)}/prepare`, { method: 'POST' });
    await refreshProfiles(profileId);
    await refreshStatus();
    state.referenceDeliveryProfile = prepared?.delivery_profile && typeof prepared.delivery_profile === 'object'
      ? { profileId, data: prepared.delivery_profile } : null;
    referenceGuidance(state.selected?.duration_seconds);
    status('Voice prepared. Reference conditioning is cached for generation.');
  } catch (error) { status(explain(error), true); }
  finally { state.saving = false; updateActions(); }
}

async function createPlayback(onError) {
  if (state.playback) {
    state.playback.node.disconnect();
    await state.playback.context.close();
  }
  const context = new AudioContext();
  await context.audioWorklet.addModule('/static/playback-worklet.js');
  await context.resume();
  const node = new AudioWorkletNode(context, 'stream-playback', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1] });
  node.connect(context.destination);
  const playback = { context, node, started: performance.now(), expected: 0, pending: null, frames: 0, sampleRate: null, cancelled: false, terminal: false, drained: false };
  state.playback = playback;
  node.onprocessorerror = () => onError('The audio playback processor stopped. Please try again.');
  node.port.onmessage = ({ data }) => {
    if (state.playback !== playback || playback.cancelled) return;
    if (data.type === 'started') {
      // This is an estimate of device output, not a microphone-measured latency.
      const deviceDelay = (context.outputLatency || 0) + (context.baseLatency || 0);
      const audible = performance.now() + (data.context_time - context.currentTime + deviceDelay) * 1000;
      const elapsed = Math.max(0, audible - playback.started) / 1000;
      $('metric-audible').textContent = `${elapsed.toFixed(2)} s`;
    }
    if (data.type === 'underrun') $('metric-gaps').textContent = String(data.count);
    if (data.type === 'drained') {
      playback.drained = true;
      void context.suspend();
      $('metric-gaps').textContent = String(data.underruns);
      if (playback.terminal) status('Speech ready.');
      updateActions();
    }
    if (data.type === 'error') onError(data.message);
  };
  return playback;
}

function stopGeneration() {
  const generating = state.generating;
  if (state.run) state.run.cancelled = true;
  if (state.playback) {
    state.playback.cancelled = true;
    state.playback.pending = null;
    state.playback.node.port.postMessage({ type: 'reset' });
    void state.playback.context.suspend();
  }
  if (!generating) {
    status('Playback stopped.');
    updateActions();
    return;
  }
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify({ type: 'cancel' }));
  else state.socket?.close();
  status('Stopping after the current phrase…');
  $('stop-button').disabled = true;
}

async function generate() {
  if (state.generating || state.capturing || state.saving || state.draft || !state.selected || !textValue().trim()) return;
  if (state.selected.duration_seconds > state.maxReferenceSeconds) { status(`Choose a reference of ${state.maxReferenceSeconds} seconds or less before generating.`, true); return; }
  const seed = Number($('speech-seed').value);
  const speed = state.speedControlEnabled ? Number($('speech-speed').value) : 1;
  const sentenceGap = Number($('sentence-gap').value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295 || !Number.isFinite(speed) || speed < 0.7 || speed > 1.4) {
    status('Use a whole-number seed from 0 to 4294967295 and a speed from 0.7× to 1.4×.', true);
    return;
  }
  if (!Number.isFinite(sentenceGap) || sentenceGap < 0 || sentenceGap > 2) {
    status('Use a sentence pause between 0 and 2 seconds.', true);
    return;
  }
  const submitted = performance.now();
  const request = { profile_id: state.selected.id, text: textValue().trim(), speed, seed, sentence_gap: sentenceGap };
  const run = { cancelled: false };
  state.run = run;
  state.generating = true; updateActions();
  $('output-player').pause(); $('output-section').hidden = true;
  for (const metric of ['audible', 'generation', 'rtf']) $(`metric-${metric}`).textContent = '—';
  $('metric-gaps').textContent = '0';
  status('Connecting to the voice engine…');
  let terminal = false;
  let playback;
  const fail = (message) => {
    terminal = true;
    if (playback) { playback.cancelled = true; playback.node.port.postMessage({ type: 'reset' }); }
    status(message, true);
    if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify({ type: 'cancel' }));
    state.socket?.close();
  };
  try {
    playback = await createPlayback(fail);
    playback.started = submitted;
    if (run.cancelled) {
      playback.cancelled = true;
      playback.node.port.postMessage({ type: 'reset' });
      state.generating = false;
      status('Generation stopped.');
      updateActions();
      return;
    }
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/generate`);
    state.socket = socket;
    socket.binaryType = 'arraybuffer';
    socket.onopen = () => {
      if (run.cancelled) socket.close();
      else socket.send(JSON.stringify(request));
    };
    socket.onmessage = ({ data }) => {
      if (terminal && playback.cancelled) return;
      try {
        if (typeof data === 'string') {
          const message = JSON.parse(data);
          if (message.type === 'audio') {
            if (playback.cancelled) return;
            if (playback.pending || !Number.isInteger(message.frames) || message.frames <= 0 || message.sequence !== playback.expected) throw new Error('The audio stream arrived out of order.');
            if (!Number.isInteger(message.sample_rate) || message.sample_rate < 8000 || message.sample_rate > 192000) throw new Error('Unsupported audio sample rate.');
            $('output-format').textContent = `WAV · ${Number((message.sample_rate / 1000).toFixed(2))} kHz`;
            playback.pending = message;
          } else if (message.type === 'complete') {
            if (playback.pending) throw new Error('The final audio packet was incomplete.');
            terminal = true; playback.terminal = true;
            playback.node.port.postMessage({ type: 'end' });
            const metrics = message.metrics || {};
            if (Number.isFinite(metrics.generation_seconds)) $('metric-generation').textContent = `${metrics.generation_seconds.toFixed(2)} s`;
            if (Number.isFinite(metrics.rtf)) $('metric-rtf').textContent = `${metrics.rtf.toFixed(2)}×`;
            if (message.audio_url) {
              $('output-player').src = message.audio_url;
              $('download-link').href = message.audio_url;
              $('output-section').hidden = false;
            }
            status(playback.cancelled ? 'Generation stopped.' : 'Generation complete. Playback will finish shortly.');
            state.generating = false; updateActions();
            void refreshStatus();
          } else if (message.type === 'cancelled') {
            terminal = true; playback.cancelled = true;
            playback.node.port.postMessage({ type: 'reset' });
            status('Generation stopped.');
            state.generating = false; updateActions();
          } else if (message.type === 'error') {
            fail(message.message || message.detail || 'Speech generation failed.');
          } else {
            const labels = { queued: 'Waiting for the voice engine…', transcribing: 'Transcribing the reference recording before preparing your voice…', loading: 'Loading the voice engine. The first request takes longer…', preparing: 'Preparing and caching this voice…', generating: 'Generating speech. Each phrase plays when it is ready…' };
            if (labels[message.type]) status(labels[message.type]);
          }
        } else {
          if (playback.cancelled) return;
          const header = playback.pending;
          if (!header || data.byteLength !== header.frames * 4) throw new Error('An audio packet was incomplete.');
          if (playback.sampleRate && playback.sampleRate !== header.sample_rate) throw new Error('The audio sample rate changed during generation.');
          playback.sampleRate = header.sample_rate;
          playback.frames += header.frames;
          if (playback.frames > header.sample_rate * MAX_OUTPUT_SECONDS) throw new Error('The generated output exceeded two minutes. Please use a shorter request.');
          const samples = new Float32Array(header.frames);
          const view = new DataView(data);
          for (let frame = 0; frame < samples.length; frame++) {
            const value = view.getFloat32(frame * 4, true);
            if (!Number.isFinite(value)) throw new Error('The audio stream contained invalid samples.');
            samples[frame] = value;
          }
          playback.node.port.postMessage({ type: 'audio', samples, sample_rate: header.sample_rate, sequence: header.sequence }, [samples.buffer]);
          playback.expected++;
          playback.pending = null;
        }
      } catch (error) { fail(explain(error)); }
    };
    socket.onerror = () => { if (!run.cancelled) fail('The voice engine connection failed. Check that the studio is running.'); };
    socket.onclose = () => {
      if (!terminal && run.cancelled) { terminal = true; status('Generation stopped.'); }
      else if (!terminal) fail('The connection closed before generation completed.');
      if (state.socket === socket) { state.socket = null; state.generating = false; updateActions(); }
    };
  } catch (error) {
    fail(explain(error));
    state.generating = false; updateActions();
  }
}

function drawPlot(canvas, x, y, { color = '#71865e', minY, maxY, xLabel = '', yLabel = '', logX = false } = {}) {
  const width = Math.max(100, canvas.clientWidth);
  const height = Math.max(80, canvas.clientHeight);
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
  const context = canvas.getContext('2d'); context.scale(scale, scale);
  const left = 43, top = 12, right = width - 5, bottom = height - 24;
  context.strokeStyle = '#e4e9dc'; context.lineWidth = 1;
  for (let i = 0; i <= 2; i++) {
    const pos = top + (bottom - top) * i / 2;
    context.beginPath(); context.moveTo(left, pos); context.lineTo(right, pos); context.stroke();
  }
  const values = y.filter(Number.isFinite);
  if (!x.length || !values.length) {
    context.fillStyle = '#7a876d'; context.font = '12px system-ui'; context.fillText('No voiced data', left, height / 2); return;
  }
  const transform = (value) => logX ? Math.log10(Math.max(value, 20)) : value;
  const lowX = transform(x[0]); const highX = transform(x[x.length - 1]);
  const lowY = minY ?? Math.min(...values); const highY = maxY ?? Math.max(...values);
  const spanX = highX - lowX || 1; const spanY = highY - lowY || 1;
  context.strokeStyle = color; context.lineWidth = 1.4;
  context.beginPath(); let pen = false;
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    if (!Number.isFinite(y[i]) || !Number.isFinite(x[i])) { pen = false; continue; }
    const px = left + (transform(x[i]) - lowX) / spanX * (right - left);
    const py = bottom - (y[i] - lowY) / spanY * (bottom - top);
    if (pen) context.lineTo(px, py); else context.moveTo(px, py);
    pen = true;
  }
  context.stroke(); context.fillStyle = '#7a876d'; context.font = '12px system-ui';
  context.fillText(`${Math.round(highY)}${yLabel}`, 0, top + 3);
  context.fillText(`${Math.round(lowY)}`, 0, bottom + 3);
  context.fillText(xLabel, left, height - 5);
}

function drawAnalysis(analysis) {
  const waveform = analysis.waveform || {};
  const pitch = analysis.pitch || {};
  const spectrum = analysis.spectrum || {};
  const voiced = (pitch.f0_hz || []).filter(Number.isFinite);
  const pitchLow = voiced.length ? Math.max(0, Math.floor((Math.min(...voiced) - 30) / 25) * 25) : 50;
  const pitchHigh = voiced.length ? Math.max(pitchLow + 100, Math.ceil((Math.max(...voiced) + 30) / 25) * 25) : 500;
  drawPlot($('waveform-plot'), waveform.time_seconds || [], waveform.amplitude || [], { minY: -1, maxY: 1, xLabel: `0 – ${(analysis.duration_seconds || state.selected?.duration_seconds || 0).toFixed(1)} s` });
  drawPlot($('pitch-plot'), pitch.time_seconds || [], pitch.f0_hz || [], { color: '#9b8358', minY: pitchLow, maxY: pitchHigh, xLabel: 'Time (s)', yLabel: ' Hz' });
  drawPlot($('spectrum-plot'), spectrum.frequency_hz || [], spectrum.relative_db || [], { color: '#6f8b9a', maxY: 0, minY: -80, xLabel: 'Frequency (Hz, log scale)', yLabel: ' dB', logX: true });
}

$('record-button').addEventListener('click', () => state.capturing ? stopCapture() : void startCapture());
$('reference-file').addEventListener('change', () => {
  const file = $('reference-file').files[0];
  if (!file) return;
  if (file.size > 16 * 1024 * 1024) { status('Use a WAV reference smaller than 16 MB and no longer than 30 seconds. Trim the kept speech to fit your voice engine.', true); return; }
  if (!/\.wav$/i.test(file.name)) { status('Upload a WAV recording.', true); return; }
  setDraft(file, file.name);
});
$('reference-player').addEventListener('loadedmetadata', () => {
  if (!state.draft || $('reference-player').src !== state.draftUrl || !Number.isFinite($('reference-player').duration)) return;
  state.draft.duration = $('reference-player').duration;
  referenceGuidance(draftReferenceDuration());
  updateActions();
});
$('profile-select').addEventListener('change', () => selectProfile($('profile-select').value));
$('voice-engine').addEventListener('change', () => void switchBackend());
$('save-reference').addEventListener('click', () => void saveReference());
$('preview-reference').addEventListener('click', () => void previewReference());
$('edit-reference').addEventListener('click', () => void editReference());
for (const id of ['trim-start', 'trim-end', 'auto-trim', 'noise-reduction']) {
  $(id).addEventListener('input', () => invalidatePreview(id !== 'noise-reduction'));
}
$('reset-reference').addEventListener('click', () => {
  $('trim-start').value = '0'; $('trim-end').value = '';
  $('noise-reduction').checked = true; $('auto-trim').checked = false;
  invalidatePreview(true);
});
$('compare-reference').addEventListener('click', () => {
  const original = $('reference-player').src === state.draftUrl;
  $('reference-player').src = original ? state.previewUrl : state.draftUrl;
  $('compare-reference').textContent = original ? 'Listen to original' : 'Listen to edited preview';
});
$('save-transcript').addEventListener('click', () => void saveTranscript());
$('prepare-voice').addEventListener('click', () => void prepareVoice());
$('generate-button').addEventListener('click', () => void generate());
$('stop-button').addEventListener('click', stopGeneration);
for (const button of document.querySelectorAll('[data-example]')) button.addEventListener('click', () => {
  $('speech-text').value = button.dataset.example;
  $('speech-text').dispatchEvent(new Event('input'));
  $('speech-text').focus();
});
for (const button of document.querySelectorAll('[data-open-help]')) button.addEventListener('click', () => $('help-dialog').showModal());
$('close-help').addEventListener('click', () => $('help-dialog').close());
for (const link of document.querySelectorAll('.nav-item')) link.addEventListener('click', () => {
  for (const item of document.querySelectorAll('.nav-item')) item.classList.toggle('active', item === link);
});
$('speech-speed').addEventListener('input', () => {
  if (state.speedControlEnabled) state.lastAdjustableSpeed = $('speech-speed').value;
  else $('speech-speed').value = '1';
  renderSpeed();
});
$('sentence-gap').addEventListener('input', renderSentenceGap);
$('speech-text').addEventListener('input', () => { $('character-count').textContent = `${textValue().length.toLocaleString()} / 3,000`; updateActions(); });
document.addEventListener('keydown', (event) => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { event.preventDefault(); void generate(); } });
window.addEventListener('pagehide', () => {
  state.socket?.close();
  state.capture?.stream.getTracks().forEach((track) => track.stop());
  if (state.playback) void state.playback.context.close();
});
new ResizeObserver(() => { if (state.selected && !$('analysis-panel').hidden) drawAnalysis(state.selected.analysis || {}); }).observe($('analysis-panel'));
$('character-count').textContent = `${textValue().length.toLocaleString()} / 3,000`;
renderSentenceGap();
updateActions();
void refreshStatus();
refreshProfiles().catch((error) => status(explain(error), true));

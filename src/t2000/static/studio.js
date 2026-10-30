const $ = (id) => document.getElementById(id);
const MAX_REFERENCE_SECONDS = 30;
const MAX_OUTPUT_SECONDS = 120;
const state = {
  profiles: [], selected: null, draft: null, draftUrl: null,
  capture: null, capturing: false, saving: false,
  socket: null, generating: false, playback: null, run: null,
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
  $('generate-button').disabled = state.generating || state.capturing || state.saving || Boolean(state.draft) || !state.selected || !textValue().trim();
  $('stop-button').disabled = !(state.generating || (state.playback && !state.playback.drained && !state.playback.cancelled));
  $('save-reference').disabled = !state.draft || state.capturing || state.saving || state.generating;
  $('save-transcript').disabled = state.generating || state.saving;
  $('prepare-voice').disabled = state.generating || state.saving || state.capturing;
  $('profile-select').disabled = state.capturing || state.saving || state.generating;
  $('reference-file').disabled = state.capturing || state.saving || state.generating;
  $('record-button').disabled = state.saving || state.generating;
}

function textValue() { return $('speech-text').value; }

async function refreshStatus() {
  try {
    const runtime = await api('/api/status');
    const ready = Boolean(runtime.loaded);
    $('runtime-status').classList.toggle('ready', ready);
    $('runtime-status').replaceChildren(Object.assign(document.createElement('i'), { ariaHidden: 'true' }), document.createTextNode(ready ? 'Voice engine ready' : 'Voice engine loads on demand'));
  } catch {
    $('runtime-status').classList.remove('ready');
    $('runtime-status').textContent = 'Studio disconnected';
  }
}

async function refreshProfiles(preferred) {
  const profiles = await api('/api/profiles');
  state.profiles = Array.isArray(profiles) ? profiles : profiles.profiles;
  $('profile-select').replaceChildren();
  if (!state.profiles.length) {
    const option = new Option('No voices yet', '');
    $('profile-select').append(option);
    state.selected = null;
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
  state.draftUrl = null;
  state.draft = null;
  $('reference-file').value = '';
  $('draft-fields').hidden = true;
  $('transcribe-label').hidden = true;
  $('save-reference').hidden = true;
}

function selectProfile(id) {
  const profile = state.profiles.find((item) => item.id === id);
  if (!profile) return;
  clearDraft();
  state.selected = profile;
  rememberProfile(id);
  $('reference-transcript').value = profile.transcript || '';
  $('save-transcript').hidden = false;
  $('prepare-voice').hidden = false;
  $('reference-player').src = `/api/profiles/${encodeURIComponent(id)}/audio`;
  $('reference-player').hidden = false;
  $('capture-status').textContent = 'Record or upload a new sample to add another voice.';
  $('reference-summary').hidden = false;
  $('reference-summary').textContent = `${profile.duration_seconds.toFixed(1)} seconds · ${(profile.sample_rate / 1000).toFixed(1)} kHz · reference cached`;
  $('analysis-panel').hidden = false;
  drawAnalysis(profile.analysis || {});
  status('Ready when you are.');
  updateActions();
}

function setDraft(blob, filename, sampleRate, duration) {
  clearDraft();
  state.draft = { blob, filename };
  state.draftUrl = URL.createObjectURL(blob);
  $('reference-player').src = state.draftUrl;
  $('reference-player').hidden = false;
  $('draft-fields').hidden = false;
  $('voice-name').value = filename.replace(/\.wav$/i, '') === 'recording' ? 'My voice' : filename.replace(/\.wav$/i, '').slice(0, 80);
  $('reference-transcript').value = '';
  $('transcribe-label').hidden = false;
  $('save-reference').hidden = false;
  $('save-transcript').hidden = true;
  $('prepare-voice').hidden = true;
  $('reference-summary').hidden = true;
  $('analysis-panel').hidden = true;
  $('capture-status').textContent = sampleRate ? `${duration.toFixed(1)} seconds captured at ${(sampleRate / 1000).toFixed(1)} kHz. Listen, then save your reference.` : `${filename} · Listen, then save your reference.`;
  status('Save this reference to use it for generation.');
  updateActions();
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
  state.capturing = true;
  updateActions();
  $('record-label').textContent = 'Starting…';
  let context, stream;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) throw new Error('Microphone recording requires a browser with AudioWorklet on localhost or HTTPS.');
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
    context = new AudioContext(); // Preserve the device/context native sample rate.
    await context.audioWorklet.addModule('/static/playback-worklet.js');
    await context.resume();
    const node = new AudioWorkletNode(context, 'reference-capture', { processorOptions: { maxFrames: context.sampleRate * MAX_REFERENCE_SECONDS }, numberOfOutputs: 1, outputChannelCount: [1] });
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
        if (capture.frames + data.samples.length > context.sampleRate * MAX_REFERENCE_SECONDS) {
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
      const elapsed = Math.min((performance.now() - capture.started) / 1000, MAX_REFERENCE_SECONDS);
      $('capture-status').textContent = `Recording ${elapsed.toFixed(1)} / ${MAX_REFERENCE_SECONDS} seconds. Click Stop recording when finished.`;
      if (elapsed >= MAX_REFERENCE_SECONDS) stopCapture();
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
  if (!state.draft || state.saving || state.capturing) return;
  state.saving = true; updateActions();
  status($('auto-transcribe').checked ? 'Saving and transcribing the reference…' : 'Analyzing and saving the reference…');
  try {
    const data = new FormData();
    data.append('file', state.draft.blob, state.draft.filename);
    data.append('name', $('voice-name').value.trim() || 'My voice');
    data.append('transcript', $('reference-transcript').value.trim());
    data.append('transcribe', String($('auto-transcribe').checked));
    const profile = await api('/api/profiles', { method: 'POST', body: data });
    await refreshProfiles(profile.id);
    status(profile.transcription_error ? `Reference saved. Transcription unavailable: ${profile.transcription_error}` : 'Reference saved. Your voice is ready to use.', Boolean(profile.transcription_error));
  } catch (error) { status(explain(error), true); }
  finally { state.saving = false; updateActions(); }
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
  state.saving = true; updateActions();
  status('Preparing this voice. The first model load can take a while…');
  try {
    await api(`/api/profiles/${encodeURIComponent(state.selected.id)}/prepare`, { method: 'POST' });
    status('Voice prepared. Reference conditioning is cached for generation.');
    await refreshStatus();
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
  const seed = Number($('speech-seed').value);
  const speed = Number($('speech-speed').value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295 || !Number.isFinite(speed) || speed < 0.7 || speed > 1.4) {
    status('Use a whole-number seed from 0 to 4294967295 and a speed from 0.7× to 1.4×.', true);
    return;
  }
  const submitted = performance.now();
  const request = { profile_id: state.selected.id, text: textValue().trim(), speed, seed };
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
            const labels = { queued: 'Waiting for the voice engine…', loading: 'Loading the voice engine. The first request takes longer…', preparing: 'Preparing and caching this voice…', generating: 'Generating speech. Each phrase plays when it is ready…' };
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

function drawPlot(canvas, x, y, { color = '#b8de96', minY, maxY, xLabel = '', yLabel = '', logX = false } = {}) {
  const width = Math.max(100, canvas.clientWidth);
  const height = Math.max(80, canvas.clientHeight);
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
  const context = canvas.getContext('2d'); context.scale(scale, scale);
  const left = 43, top = 12, right = width - 5, bottom = height - 24;
  context.strokeStyle = '#303a31'; context.lineWidth = 1;
  for (let i = 0; i <= 2; i++) {
    const pos = top + (bottom - top) * i / 2;
    context.beginPath(); context.moveTo(left, pos); context.lineTo(right, pos); context.stroke();
  }
  const values = y.filter(Number.isFinite);
  if (!x.length || !values.length) {
    context.fillStyle = '#a6b2a7'; context.font = '12px system-ui'; context.fillText('No voiced data', left, height / 2); return;
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
  context.stroke(); context.fillStyle = '#a5b3a7'; context.font = '12px system-ui';
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
  drawPlot($('pitch-plot'), pitch.time_seconds || [], pitch.f0_hz || [], { color: '#e0c78a', minY: pitchLow, maxY: pitchHigh, xLabel: 'Time (s)', yLabel: ' Hz' });
  drawPlot($('spectrum-plot'), spectrum.frequency_hz || [], spectrum.relative_db || [], { color: '#9bbad6', maxY: 0, minY: -80, xLabel: 'Frequency (Hz, log scale)', yLabel: ' dB', logX: true });
}

$('record-button').addEventListener('click', () => state.capturing ? stopCapture() : void startCapture());
$('reference-file').addEventListener('change', () => {
  const file = $('reference-file').files[0];
  if (!file) return;
  if (file.size > 16 * 1024 * 1024) { status('Use a WAV reference smaller than 16 MB and shorter than 30 seconds.', true); return; }
  if (!/\.wav$/i.test(file.name)) { status('Upload a WAV recording.', true); return; }
  setDraft(file, file.name);
});
$('profile-select').addEventListener('change', () => selectProfile($('profile-select').value));
$('save-reference').addEventListener('click', () => void saveReference());
$('save-transcript').addEventListener('click', () => void saveTranscript());
$('prepare-voice').addEventListener('click', () => void prepareVoice());
$('generate-button').addEventListener('click', () => void generate());
$('stop-button').addEventListener('click', stopGeneration);
$('speech-speed').addEventListener('input', () => { $('speed-value').value = `${Number($('speech-speed').value).toFixed(2).replace(/0$/, '')}×`; });
$('speech-text').addEventListener('input', () => { $('character-count').textContent = `${textValue().length.toLocaleString()} / 3,000`; updateActions(); });
document.addEventListener('keydown', (event) => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { event.preventDefault(); void generate(); } });
window.addEventListener('pagehide', () => {
  state.socket?.close();
  state.capture?.stream.getTracks().forEach((track) => track.stop());
  if (state.playback) void state.playback.context.close();
});
new ResizeObserver(() => { if (state.selected && !$('analysis-panel').hidden) drawAnalysis(state.selected.analysis || {}); }).observe($('analysis-panel'));
$('character-count').textContent = `${textValue().length.toLocaleString()} / 3,000`;
updateActions();
void refreshStatus();
refreshProfiles().catch((error) => status(explain(error), true));

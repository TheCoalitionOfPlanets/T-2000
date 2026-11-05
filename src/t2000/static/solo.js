/* A single-voice client: the server pins the voice, so a request carries only words. */
const $ = (id) => document.getElementById(id);
const MAX_OUTPUT_SECONDS = 600;
const state = {
  voice: null, socket: null, generating: false, playback: null, run: null,
  speedControlEnabled: true, warm: null,
};

function explain(error) { return error?.message || String(error); }

function status(message, error = false) {
  $('session-status').textContent = message;
  $('session-status').classList.toggle('error', error);
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
    } catch { /* The HTTP status remains actionable without a JSON body. */ }
    throw new Error(message);
  }
  return response.json();
}

function updateActions() {
  const ready = Boolean(state.voice);
  $('speak-button').disabled = state.generating || !ready || !$('speech-text').value.trim();
  $('stop-button').disabled = !(state.generating || (state.playback && !state.playback.drained && !state.playback.cancelled));
  $('speech-speed').disabled = !state.speedControlEnabled;
  if (!state.speedControlEnabled) { $('speech-speed').value = '1'; renderSpeed(); }
  $('listening-room').setAttribute('aria-busy', String(state.generating));
}

function renderSpeed() {
  $('speed-value').value = `${Number($('speech-speed').value).toFixed(2).replace(/0$/, '')}×`;
}

function renderCount() {
  const limit = state.voice?.max_text_chars || 3000;
  $('character-count').textContent = `${$('speech-text').value.length.toLocaleString()} / ${limit.toLocaleString()}`;
}

function renderWarm(warm) {
  const element = $('warm-state');
  element.classList.toggle('ready', warm?.state === 'ready');
  element.classList.toggle('error', Boolean(warm?.detail));
  const labels = {
    cold: warm?.detail ? `Not ready — ${warm.detail}` : 'Preparing this voice…',
    warming: 'Warming this voice…',
    ready: warm?.seconds ? `Ready · warmed in ${warm.seconds.toFixed(1)} s` : 'Ready',
  };
  element.lastChild.textContent = labels[warm?.state] || 'Connecting…';
}

async function refreshVoice() {
  const voice = await api('/api/voice');
  state.voice = voice;
  state.speedControlEnabled = voice.speed_control?.enabled !== false;
  state.warm = voice.warm;
  document.title = voice.title;
  $('app-title').textContent = voice.title;
  $('voice-name').textContent = voice.voice;
  $('voice-detail').textContent = `${voice.duration_seconds.toFixed(1)} s reference`;
  $('engine-name').textContent = voice.backend;
  $('speech-text').maxLength = voice.max_text_chars;
  renderWarm(voice.warm);
  renderCount();
  updateActions();
  return voice;
}

async function pollUntilWarm() {
  for (let attempt = 0; attempt < 600; attempt++) {
    const voice = await refreshVoice().catch(() => null);
    if (!voice) return;
    if (voice.warm.state !== 'warming') {
      status(voice.warm.state === 'ready'
        ? 'This voice is ready. Write something and press Speak.'
        : voice.warm.detail || 'The voice will be prepared on your first request.',
        voice.warm.state !== 'ready' && Boolean(voice.warm.detail));
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
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
      $('metric-audible').textContent = `${(Math.max(0, audible - playback.started) / 1000).toFixed(2)} s`;
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

async function speak() {
  if (state.generating || !state.voice || !$('speech-text').value.trim()) return;
  const seed = Number($('speech-seed').value);
  const speed = state.speedControlEnabled ? Number($('speech-speed').value) : 1;
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295 || !Number.isFinite(speed) || speed < 0.7 || speed > 1.4) {
    status('Use a whole-number seed from 0 to 4294967295 and a speed from 0.7× to 1.4×.', true);
    return;
  }
  const submitted = performance.now();
  const request = { text: $('speech-text').value.trim(), speed, seed };
  const run = { cancelled: false };
  state.run = run;
  state.generating = true; updateActions();
  $('output-player').pause(); $('output-player').hidden = true; $('download-link').hidden = true;
  for (const metric of ['audible', 'generation', 'rtf']) $(`metric-${metric}`).textContent = '—';
  $('metric-gaps').textContent = '0';
  status('Connecting to the voice…');
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
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/speak`);
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
              $('output-player').hidden = false;
              $('download-link').href = message.audio_url;
              $('download-link').hidden = false;
            }
            status(playback.cancelled ? 'Generation stopped.' : 'Generation complete. Playback will finish shortly.');
            state.generating = false; updateActions();
            void refreshVoice().catch(() => {});
          } else if (message.type === 'cancelled') {
            terminal = true; playback.cancelled = true;
            playback.node.port.postMessage({ type: 'reset' });
            status('Generation stopped.');
            state.generating = false; updateActions();
          } else if (message.type === 'error') {
            fail(message.message || message.detail || 'Speech generation failed.');
          } else {
            const labels = { queued: 'Waiting for the voice…', transcribing: 'Transcribing the reference recording…', loading: 'Loading the voice engine. The first request takes longer…', preparing: 'Preparing this voice…', generating: 'Speaking. Each phrase plays when it is ready…' };
            if (labels[message.type]) status(labels[message.type]);
          }
        } else {
          if (playback.cancelled) return;
          const header = playback.pending;
          if (!header || data.byteLength !== header.frames * 4) throw new Error('An audio packet was incomplete.');
          if (playback.sampleRate && playback.sampleRate !== header.sample_rate) throw new Error('The audio sample rate changed during generation.');
          playback.sampleRate = header.sample_rate;
          playback.frames += header.frames;
          if (playback.frames > header.sample_rate * MAX_OUTPUT_SECONDS) throw new Error('The generated output exceeded the output limit. Please use shorter text.');
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
    socket.onerror = () => { if (!run.cancelled) fail('The voice connection failed. Check that the app is running.'); };
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

$('speak-button').addEventListener('click', () => void speak());
$('stop-button').addEventListener('click', stopGeneration);
$('speech-speed').addEventListener('input', renderSpeed);
$('speech-text').addEventListener('input', () => { renderCount(); updateActions(); });
document.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { event.preventDefault(); void speak(); }
});

renderSpeed();
renderCount();
refreshVoice().then(pollUntilWarm).catch((error) => status(explain(error), true));

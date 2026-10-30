#!/usr/bin/env node
/** Dependency-free checks for the actual browser AudioWorklet implementation.
 * Optional UI smoke: T2000_BROWSER_URL=http://127.0.0.1:7861
 * PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/check_browser.mjs
 * Set T2000_BROWSER_INTEGRATION=1 only against a server with an injected test
 * backend: this uploads a reference, prepares conditioning, and generates audio.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import { pathToFileURL } from 'node:url';

const processors = new Map();
class AudioWorkletProcessor {
  constructor() {
    this.messages = [];
    this.port = { postMessage: (message) => this.messages.push(message), onmessage: null };
  }
}
const scope = vm.createContext({
  AudioWorkletProcessor, Float32Array, Math, Number, sampleRate: 48000, currentTime: 0,
  registerProcessor: (name, cls) => processors.set(name, cls),
});
vm.runInContext(await fs.readFile(new URL('../src/t2000/static/playback-worklet.js', import.meta.url), 'utf8'), scope);
const Capture = processors.get('reference-capture');
const Playback = processors.get('stream-playback');

const capture = new Capture({ processorOptions: { maxFrames: 5 } });
const captureOutputs = [[new Float32Array(128)]];
capture.process([[new Float32Array([1, 0, -1]), new Float32Array([0, 0, 0])]], captureOutputs);
capture.process([[new Float32Array([0.25, 0.5, 0.75])]], captureOutputs);
assert.equal(capture.messages.length, 2, 'final partial capture batch precedes the completion acknowledgement');
assert.deepEqual([...capture.messages[0].samples], [0.5, 0, -0.5, 0.25, 0.5]);
assert.equal(capture.messages[1].frames, 5, 'the worklet caps native frames exactly');
assert.equal(capture.messages[1].type, 'stopped');
capture.port.onmessage({ data: { type: 'stop' } });
assert.equal(capture.messages.length, 2, 'finalization is idempotent');

const manualCapture = new Capture({ processorOptions: { maxFrames: 48000 } });
manualCapture.process([[new Float32Array([0.2, 0.3, 0.4])]], captureOutputs);
manualCapture.port.onmessage({ data: { type: 'stop' } });
assert.equal(manualCapture.messages[0].samples.length, 3, 'manual stop preserves the partial batch');
assert.equal(manualCapture.messages[1].frames, 3);

const playback = new Playback();
playback.port.onmessage({ data: { type: 'audio', sequence: 0, sample_rate: 24000, samples: new Float32Array([0, 1, 0, -1]) } });
playback.port.onmessage({ data: { type: 'end' } });
const output = [[new Float32Array(128)]];
playback.process([], output);
assert.deepEqual([...output[0][0].slice(0, 8)], [0, 0.5, 1, 0.5, 0, -0.5, -1, -1], '24 kHz audio is played at its original pitch on a 48 kHz device');
assert.equal(playback.messages.filter((message) => message.type === 'started').length, 1);
assert.equal(playback.messages.find((message) => message.type === 'drained').underruns, 0);
assert.equal(playback.blocks[0], null, 'consumed blocks release their sample arrays immediately');

const gaps = new Playback();
gaps.process([], output);
assert.equal(gaps.underruns, 0, 'startup buffering is not an underrun');
gaps.port.onmessage({ data: { type: 'audio', sequence: 0, sample_rate: 48000, samples: new Float32Array(256).fill(0.25) } });
gaps.process([], output); gaps.process([], output); gaps.process([], output);
assert.equal(gaps.underruns, 1, 'a continued starvation period is counted only once');
gaps.port.onmessage({ data: { type: 'audio', sequence: 1, sample_rate: 48000, samples: new Float32Array(256).fill(0.25) } });
gaps.process([], output); gaps.process([], output); gaps.process([], output);
assert.equal(gaps.underruns, 2, 'a later gap is counted separately');

const sequence = new Playback();
sequence.port.onmessage({ data: { type: 'audio', sequence: 1, sample_rate: 48000, samples: new Float32Array([1]) } });
assert.equal(sequence.messages[0].type, 'error', 'out-of-order packets fail instead of corrupting playback');
const limit = new Playback();
limit.port.onmessage({ data: { type: 'audio', sequence: 0, sample_rate: 8000, samples: new Float32Array(8000 * 120 + 1) } });
assert.equal(limit.messages[0].type, 'error', 'playback memory is bounded');
console.log('AudioWorklet checks passed: capture finalization, native rates, resampling, gaps, ordering, and memory bounds.');

if (process.env.T2000_BROWSER_URL) {
  const playwrightPath = process.env.PLAYWRIGHT_MODULE;
  if (!playwrightPath) throw new Error('Set PLAYWRIGHT_MODULE to the installed playwright/index.mjs path for the optional browser check.');
  const { chromium } = await import(pathToFileURL(playwrightPath).href);
  const browser = await chromium.launch({ headless: true, args: ['--autoplay-policy=no-user-gesture-required', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.T2000_BROWSER_URL);
    await page.getByRole('heading', { name: 'Make it sound like you.' }).waitFor();
    await page.waitForFunction(() => !document.getElementById('runtime-status').textContent.includes('Connecting'));
    assert.equal(await page.getByLabel('Text to generate').inputValue(), 'Hello. This is my voice, with something new to say.');
    if (process.env.T2000_BROWSER_INTEGRATION === '1') {
      const rate = 48000, frames = rate * 2;
      const wav = Buffer.alloc(44 + frames * 4);
      wav.write('RIFF', 0); wav.writeUInt32LE(36 + frames * 4, 4); wav.write('WAVEfmt ', 8);
      wav.writeUInt32LE(16, 16); wav.writeUInt16LE(3, 20); wav.writeUInt16LE(1, 22);
      wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 4, 28);
      wav.writeUInt16LE(4, 32); wav.writeUInt16LE(32, 34); wav.write('data', 36);
      wav.writeUInt32LE(frames * 4, 40);
      for (let i = 0; i < frames; i++) wav.writeFloatLE(0.18 * Math.sin(2 * Math.PI * 173 * i / rate), 44 + i * 4);
      await page.locator('#reference-file').setInputFiles({ name: 'browser-check.wav', mimeType: 'audio/wav', buffer: wav });
      await page.locator('#voice-name').fill('Browser transport check');
      await page.locator('#save-reference').click();
      await page.locator('#session-status').filter({ hasText: 'Reference saved.' }).waitFor();
      assert.equal(await page.locator('#analysis-panel').isVisible(), true);
      await page.locator('#reference-transcript').fill('Reference for transport checks.');
      await page.locator('#save-transcript').click();
      await page.locator('#session-status').filter({ hasText: 'Transcript saved.' }).waitFor();
      await page.locator('#prepare-voice').click();
      await page.locator('#session-status').filter({ hasText: 'Voice prepared.' }).waitFor();
      await page.locator('#speech-text').fill('Hello there! This is a second phrase.');
      await page.locator('#generate-button').click();
      await page.locator('#output-section').waitFor({ state: 'visible' });
      await page.waitForFunction(() => document.getElementById('metric-audible').textContent !== '—');
      assert.match(await page.locator('#metric-generation').textContent(), /\d+\.\d+ s/);
      assert.match(await page.locator('#metric-rtf').textContent(), /\d+\.\d+×/);
      const output = new URL(await page.locator('#download-link').getAttribute('href'), page.url());
      assert.equal((await page.request.get(output.href)).status(), 200);
      await page.setViewportSize({ width: 390, height: 844 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, 'mobile layout must not overflow');
      await page.locator('#record-button').click();
      await page.locator('#record-label').filter({ hasText: 'Stop recording' }).waitFor();
      await page.waitForTimeout(1200);
      await page.locator('#record-button').click();
      await page.locator('#capture-status').filter({ hasText: 'seconds captured at' }).waitFor();
      assert.equal(await page.locator('#generate-button').isDisabled(), true, 'unsaved references must not generate using an old voice');
      console.log('Browser integration passed: WAV enrollment, graphs, transcript edit, preparation, streaming playback, audible timing, download, mobile layout, microphone finalization, and draft isolation. Test-double timings are not inference benchmarks.');
    }
    assert.equal(errors.length, 0, errors.join('\n'));
    console.log('Browser studio loaded without JavaScript errors.');
  } finally { await browser.close(); }
}

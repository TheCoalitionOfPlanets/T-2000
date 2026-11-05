/* AudioWorklet processors run on the audio rendering thread, away from UI work. */
class ReferenceCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions || {};
    this.limit = Math.min(Math.floor(config.maxFrames || sampleRate * 30), sampleRate * 30);
    this.batch = new Float32Array(2048);
    this.used = 0;
    this.total = 0;
    this.finished = false;
    this.port.onmessage = ({ data }) => {
      if (data.type === 'stop') this.finish();
    };
  }

  flush() {
    if (!this.used) return;
    const samples = this.batch.slice(0, this.used);
    this.port.postMessage({ type: 'samples', samples }, [samples.buffer]);
    this.used = 0;
  }

  finish() {
    if (this.finished) return;
    this.finished = true;
    this.flush();
    this.port.postMessage({ type: 'stopped', frames: this.total, sample_rate: sampleRate });
  }

  process(inputs, outputs) {
    for (const channels of outputs) for (const channel of channels) channel.fill(0);
    if (this.finished) return false;
    const channels = inputs[0];
    if (!channels?.length) return true;
    const count = Math.min(channels[0].length, this.limit - this.total);
    for (let frame = 0; frame < count; frame++) {
      let value = 0;
      for (const channel of channels) value += channel[frame];
      this.batch[this.used++] = value / channels.length;
      this.total++;
      if (this.used === this.batch.length) this.flush();
    }
    if (this.total >= this.limit) this.finish();
    return !this.finished;
  }
}

class StreamPlayback extends AudioWorkletProcessor {
  constructor() {
    super();
    this.reset();
    this.port.onmessage = ({ data }) => {
      if (data.type === 'reset') this.reset();
      if (data.type === 'audio') this.enqueue(data);
      if (data.type === 'end') this.ended = true;
    };
  }

  reset() {
    this.blocks = [];
    this.blockIndex = 0;
    this.offset = 0;
    this.frames = 0;
    this.phase = 0;
    this.sourceRate = null;
    this.sequence = 0;
    this.started = false;
    this.hadAudio = false;
    this.starving = false;
    this.underruns = 0;
    this.ended = false;
    this.drained = false;
    this.failed = false;
  }

  fail(message) {
    this.failed = true;
    this.blocks = [];
    this.frames = 0;
    this.port.postMessage({ type: 'error', message });
  }

  enqueue(data) {
    if (this.failed) return;
    if (this.ended || data.sequence !== this.sequence || !(data.samples instanceof Float32Array)) {
      this.fail('The audio stream arrived out of order.');
      return;
    }
    if (!Number.isInteger(data.sample_rate) || data.sample_rate < 8000 || data.sample_rate > 192000) {
      this.fail('The audio stream has an unsupported sample rate.');
      return;
    }
    if (this.sourceRate && this.sourceRate !== data.sample_rate) {
      this.fail('The audio sample rate changed during generation.');
      return;
    }
    this.sourceRate = data.sample_rate;
    if (this.frames + data.samples.length > this.sourceRate * 600) {
      this.fail('Playback buffer exceeded the output limit. Please use shorter text.');
      return;
    }
    this.sequence++;
    if (data.samples.length) {
      this.blocks.push(data.samples);
      this.frames += data.samples.length;
    }
  }

  peek(ahead) {
    let index = this.blockIndex;
    let offset = this.offset + ahead;
    while (index < this.blocks.length && offset >= this.blocks[index].length) {
      offset -= this.blocks[index++].length;
    }
    return index < this.blocks.length ? this.blocks[index][offset] : null;
  }

  consume(count) {
    count = Math.min(count, this.frames);
    this.frames -= count;
    while (count && this.blockIndex < this.blocks.length) {
      const left = this.blocks[this.blockIndex].length - this.offset;
      if (count >= left) {
        count -= left;
        this.blocks[this.blockIndex] = null;
        this.blockIndex++;
        this.offset = 0;
      } else {
        this.offset += count;
        count = 0;
      }
    }
    // Drop consumed blocks periodically so a long stream does not retain them.
    if (this.blockIndex > 16) {
      this.blocks = this.blocks.slice(this.blockIndex);
      this.blockIndex = 0;
    }
  }

  process(_inputs, outputs) {
    const channel = outputs[0][0];
    channel.fill(0);
    if (this.failed) return true;
    const ratio = this.sourceRate ? this.sourceRate / sampleRate : 1;
    for (let frame = 0; frame < channel.length; frame++) {
      // Keep one source sample of lookahead until the producer signals EOS.
      const required = this.ended ? 1 : Math.max(2, Math.ceil(ratio));
      if (this.frames < required) {
        if (this.hadAudio && !this.ended && !this.starving) {
          this.starving = true;
          this.underruns++;
          this.port.postMessage({ type: 'underrun', count: this.underruns });
        }
        if (this.ended && !this.frames && !this.drained) {
          this.drained = true;
          this.port.postMessage({ type: 'drained', underruns: this.underruns });
        }
        continue;
      }
      this.starving = false;
      const first = this.peek(0);
      const next = this.peek(1) ?? first;
      channel[frame] = first + (next - first) * this.phase;
      this.hadAudio = true;
      if (!this.started && Math.abs(channel[frame]) > 0.00001) {
        this.started = true;
        this.port.postMessage({ type: 'started', context_time: currentTime + frame / sampleRate });
      }
      this.phase += ratio;
      const advance = Math.floor(this.phase);
      this.phase -= advance;
      this.consume(advance);
    }
    return true;
  }
}

registerProcessor('reference-capture', ReferenceCapture);
registerProcessor('stream-playback', StreamPlayback);

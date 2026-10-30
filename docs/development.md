# Development checks

```sh
uv pip install --python .venv/bin/python -c requirements/constraints.txt -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
node scripts/check_browser.mjs
```

Tests cover reference validation, cache invalidation, optional STT, adapter boundaries, packet ordering, serialization, cancellation, malformed messages and publication failures. Adapter tests replace optional upstream dependencies at their boundary; they do not establish checkpoint compatibility or quality.

The Node command checks AudioWorklet capture tails, mono mixing, resampling, ordering, underruns and buffer bounds without dependencies. Optional Chromium checks require an explicitly supplied Playwright installation and running studio:

```sh
T2000_BROWSER_URL=http://127.0.0.1:7861 \
  PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs \
  node scripts/check_browser.mjs
```

`T2000_BROWSER_INTEGRATION=1` exercises a dedicated test server created with `create_app(settings, backend=...)`. Use only an isolated data directory and injected test backend. This mode uploads a diagnostic signal and checks preparation, transport, recording and downloads. Its timings are not cloning results. There is no fake backend in the product CLI.

The studio loads Torch lazily and downloads models only through explicit setup commands. Blocking model work runs outside the event loop. A bounded transport queue and one model lock serialize inference within a process; run one process per GPU.

Reference WAVs are immutable content-addressed artifacts. Transcript edits atomically replace metadata and change conditioning cache keys. Completed WAVs become visible after generation and sidecar publication succeed. Cancellation is cooperative between model calls and packets; a running kernel is allowed to finish.

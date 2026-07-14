# Pi harness + local-LLM (gemma4) pathway

The runner drives **Pi** (`@earendil-works/pi-coding-agent`) as a harness
(`harness: "pi"`) for a lightweight local-LLM pathway: Pi makes OpenAI-style
calls to a local **gemma4** model served by ollama.

## Thinking/reasoning off — done in Pi, no proxy

gemma4 must be told to skip its thinking phase or it burns minutes with no
streamed output. Pi does this **natively** via the extension
`pi_harness_extensions/reasoning_control_and_provider_traffic_log.ts`, whose
`before_provider_request` hook injects the concrete fields gemma4 needs
(`reasoning_enabled: false`, `reasoning_effort: "none"`) into every provider
call, and logs all provider traffic for debugging.

Driven by environment (set on the VM runner serve so every pi launch inherits it):

- `PI_EXTENSION_PATHS` — `:`-separated extension files; the PiAdapter turns these
  into `-e` flags.
- `PI_REQUEST_PAYLOAD_INJECTION_JSON` — the **concrete** per-model payload merge,
  e.g. `{"reasoning_enabled": false, "reasoning_effort": "none"}`. A model that
  should keep thinking omits this. (Not a generic on/off abstraction — you state
  the literal fields the model needs.)
- `PI_PROVIDER_TRAFFIC_LOG_PATH` — provider request/response debug log.
- `SEARXNG_BASE_URL` — the `web_search` extension's SearXNG JSON API.

The full portable setup (install, `models.json`, extensions, env) is the
reference repo **[RandyHaylor/pi-gemma4](https://github.com/RandyHaylor/pi-gemma4)**.

## opencode + gemma4: needs the external proxy (not this extension)

opencode has **no per-call payload hook**, so it cannot inject
`reasoning_enabled` itself. To turn gemma4 thinking off for opencode we use a
small standalone reverse proxy that injects the field before forwarding to ollama:

- Script: `unharness-live-store/reasoning_off_injecting_ollama_proxy.py`
  (listens `:11500`, forwards to ollama `:11434`; injects `reasoning_enabled:false`
  + `reasoning_effort:"none"`; tees all traffic to a log).
- Point opencode's `baseURL` at `http://<host>:11500/v1` when using it.

This proxy is **only** needed for opencode; **pi does not use it** (the extension
replaces it). It is intentionally NOT built into the runner.

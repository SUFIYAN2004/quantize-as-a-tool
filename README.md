# Quantize-as-a-Tool

**Model compression that AI agents can call, with an accuracy budget and proof.**

An agent sends a Hugging Face model ID and an accuracy budget. It gets back the smallest 4-bit or 8-bit version of that model that stays within the budget, together with measured size, measured quality change and a SHA-256 of the artifact. It is exposed over HTTP and as an MCP tool.

> **Category:** Software built for agents. Compression tools are built for ML engineers, with many formats, silent quality loss and no verification. This project gives agents one safe, bounded, verifiable call instead.

---

## Why this exists

An agent that needs a smaller model (for an edge device, lower cost or faster inference) has no safe way to get one today. Calling a quantizer blindly can ship a degraded model without anyone noticing, and the right format differs from model to model.

This service makes the trade-off explicit:

- The caller states how much quality it can lose (`accuracy_budget_pct`).
- The service **measures** each candidate format against the fp16 baseline instead of assuming.
- The caller receives the smallest format that passes, or a clear refusal.

## Measured results

Tesla T4, WikiText-2 perplexity (40k tokens, 512-token windows), 10% budget unless noted.

| Model | Params | Format chosen | Size (vs fp16) | Perplexity change |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | int8 | 0.63 GB (vs 0.99 GB) | +0.89% |
| TinyLlama-1.1B-Chat | 1.1B | NF4 4-bit | 0.75 GB (vs 2.2 GB) | +5.42% |
| Qwen2.5-1.5B | 1.5B | NF4 4-bit | 1.12 GB (vs ~3.1 GB) | +8.66% |
| Phi-2 | 2.7B | NF4 4-bit | 1.78 GB (vs ~5.6 GB) | +5.94% |
| SmolLM2-1.7B | 1.7B | int8 | 1.81 GB (vs 3.42 GB) | +2.64% |
| OLMo-1B | 1.2B | NF4 4-bit | 0.95 GB (vs ~2.4 GB) | +7.21% |

**Key finding:** the right format depends on the model. On Qwen2.5-0.5B, 4-bit costs +12.9% perplexity and fails a 10% budget, so the service falls back to int8. With a 15% budget the same model accepts 4-bit at about 0.47 GB. On TinyLlama, Qwen2.5-1.5B, Phi-2 and OLMo-1B, 4-bit passes on the first try. SmolLM2-1.7B is a counter-example: 4-bit costs +15.3%, so the service falls back to int8 even though the model is larger than Qwen2.5-0.5B. Within the Qwen family, the 4-bit penalty shrinks as the model grows (+12.9% at 0.5B, +8.7% at 1.5B). Model size alone does not predict which format wins, so a fixed rule such as "use 4-bit above 1B parameters" would have shipped SmolLM2 with a 15% quality loss. Measuring against the budget avoids that. For SmolLM2, a 20% budget would accept 4-bit at about 1.0 GB.

Raw reports are in [`results/`](results/), and every measured configuration is cached in [`memory.json`](memory.json).

## How it works

```mermaid
flowchart LR
    A[Calling agent] -->|model_id, budget| B[Fetcher]
    B -->|safe + feasible| C[Quantizer]
    B -->|refused + flags| A
    C -->|candidates, smallest first| D{Meets budget?}
    D -->|no| C
    D -->|yes| E[Validator]
    M[(memory.json)] <--> C
    E -->|artifact, size, ppl change, SHA-256| A
```

1. **Fetcher** reads metadata only, with no weight download and no code execution. It refuses gated, pickle-only, remote-code (`trust_remote_code`), already-quantized and oversized (>3B parameters) models, and returns the reasons as machine-readable flags.
2. **Quantizer** walks a ladder of formats (NF4 + double quantization, NF4, int8), smallest first. It stops at the first one whose perplexity increase over the fp16 baseline is within the budget.
3. **Validator** records the perplexity change, the size and a SHA-256 of the saved weights so the result can be checked later.
4. **Memory** stores every measured configuration per model. Known failures are skipped and known passes are not re-measured, so repeat requests return in seconds.

## Agent-facing features

| Feature | Behavior |
|---|---|
| Explicit budget | `accuracy_budget_pct` is an input, not a hidden default |
| Clean refusals | Bad input returns `200` with `status: REFUSED` and `flags`, never a crash |
| Auth | `/quantize` requires an `X-API-Key` header |
| GPU lock | Concurrent requests queue instead of fighting over VRAM |
| Size cap | Models above 3B parameters are refused before any download |
| Plan before you spend | `POST /plan` answers "which known models fit my size and quality limits?" from the cache, with no GPU, so an agent can decide before it asks for a job |
| Async jobs | `POST /jobs` returns a job ID instantly and `GET /jobs/{id}` reports progress, so long runs never hit a tunnel timeout |
| Per-request output | Each request writes to its own artifact folder |
| Verification | Result includes size, perplexity change and artifact SHA-256 |
| Signed reports | Every `SUCCESS` or `NO_CONFIG_MET_BUDGET` result is signed with Ed25519, and the public key is served at `GET /public-key` |

## Demo: an agent picks a model for a 1 GB device

[`demo_agent.py`](demo_agent.py) is a small agent with a concrete task: *deploy a chat model to a device with 1 GB for weights, accepting at most 8% perplexity loss.* It uses only the service's HTTP API:

1. `POST /plan` asks which already-measured models fit both limits (cheap, no GPU).
2. It verifies the plan's signature before trusting it.
3. It submits the best candidate as a job and polls it.
4. It checks the signed receipt (status, signature, model, quality loss, size, artifact hash) before deciding.
5. It confirms that an edited receipt fails verification.

With the measurements in this repository, at an 8% budget and a 1 GB limit, three models fit (Qwen2.5-0.5B as int8, TinyLlama-1.1B and OLMo-1B as 4-bit), Phi-2 and SmolLM2-1.7B are rejected as too big, and Qwen2.5-1.5B is rejected as over budget. The agent picks Qwen2.5-0.5B int8 (+0.89%, 0.63 GB), because it has the lowest quality loss.

```bash
BUDGET_PCT=8 MAX_SIZE_GB=1.0 QUANTIZE_URL=http://127.0.0.1:8000 \
QUANTIZE_API_KEY=$KEY python demo_agent.py
```

The same pattern, an explicit limit, a measured result and a signed receipt the agent checks before acting, applies to other agent actions. Quantization is the first example. In the demo the agent trusts the key the service advertises. A real agent would pin the public key in advance (`TRUSTED_PUBLIC_KEY`).

## Quick start (Google Colab, T4 GPU)

Set the runtime to **T4 GPU**, then:

```python
!git clone https://github.com/SUFIYAN2004/quantize-as-a-tool
%cd quantize-as-a-tool
!pip install -q -U -r requirements.txt
%run quantize_service.py
server, key, url = start_service()
print(url)  # the API key is saved to /content/api_key.txt
```

`start_service()` starts the API on port 8000 and opens a temporary public URL through a Cloudflare quick tunnel.

## API

### `POST /quantize`

Header: `X-API-Key: <key>`

```json
{
  "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "accuracy_budget_pct": 10.0,
  "bits": "auto"
}
```

`bits` is `"auto"`, `"4"` or `"8"`.

```bash
curl -X POST "$URL/quantize" \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "accuracy_budget_pct": 10}'
```

**Success**

```json
{
  "status": "SUCCESS",
  "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "winner": "nf4+double_quant",
  "winner_stats": { "ppl_increase_pct": 5.42, "size_gb": 0.747 },
  "fp16_size_gb": 2.2,
  "attempts": [
    { "config": "nf4+double_quant", "ppl_increase_pct": 5.42, "size_gb": 0.747,
      "source": "memory", "verdict": "PASS" }
  ],
  "artifact_sha256": "…"
}
```

**Refusal**

```json
{ "status": "REFUSED", "flags": ["TOO_LARGE: 7615.6M params > 3000M limit"] }
```

Possible flags: `GATED`, `NOT_FOUND`, `TOO_LARGE`, `REMOTE_CODE`, `PICKLE_ONLY`, `ALREADY_QUANTIZED`, `FETCH_ERROR`.

Other statuses: `NO_CONFIG_MET_BUDGET` (no format stayed within the budget) and `UNAUTHORIZED` (HTTP 401, missing or wrong key).

### `POST /jobs` and `GET /jobs/{job_id}` (recommended for new models)

`POST /jobs` takes the same body as `/quantize` and returns immediately:

```json
{ "job_id": "c1cd88e92203", "status": "QUEUED" }
```

Poll `GET /jobs/{job_id}` (same `X-API-Key` header) every 10 to 30 seconds. `status` moves through `QUEUED`, `RUNNING`, then `DONE` or `FAILED`. When `DONE`, the response includes `result`, which has the same shape as the `/quantize` response (a `SUCCESS` result or a `REFUSED` one with flags). An unknown job ID returns `404`.

### `POST /plan`

Header: `X-API-Key: <key>`. Reads only the measurement cache, so it returns instantly and needs no GPU.

```json
{ "accuracy_budget_pct": 8.0, "max_size_gb": 1.0, "model_ids": ["optional", "subset"] }
```

Returns `status: "PLAN"` (signed like other reports) with `recommended`, `fits` (best quality first) and `rejected`, each rejection having a `reason` of `too_big`, `over_budget` or `not_measured`. For each model it names the config a job with the same budget would deliver. Sizes are weight memory footprints and exclude activations and the KV cache, so leave headroom on a real device. A plan covers only models that have been measured, so an `over_budget` result with unmeasured formats listed may still be satisfiable by a job.

### `GET /public-key`

Returns `{"alg": "ed25519", "public_key": "<base64>"}`. No API key required.

### `GET /health`

Returns `{"ok": true}`. No key required.

## Task benchmark check (ARC-Easy)

Perplexity is a proxy, so we spot-checked whether the budget tracks real task accuracy. Multiple-choice accuracy on the first 400 ARC-Easy test questions, scored by length-normalised log-likelihood of each answer (a small custom scorer, not `lm-eval`). Raw numbers: [`results/benchmarks_arc.json`](results/benchmarks_arc.json).

| Model | Format | Perplexity change | ARC-Easy accuracy |
|---|---|---|---|
| Qwen2.5-0.5B | fp16 | baseline | 59.2% |
| Qwen2.5-0.5B | NF4 4-bit | +12.9% (fails 10% budget) | 56.2% (-3.0) |
| Qwen2.5-0.5B | int8 | +0.9% (passes) | 59.2% (0.0) |
| TinyLlama-1.1B | fp16 | baseline | 55.2% |
| TinyLlama-1.1B | NF4 4-bit | +5.4% (passes) | 54.8% (-0.4) |

The formats that passed the perplexity budget showed no measurable accuracy change, and the one that failed showed the only visible drop. This is consistent with the budget doing its job, but it is not conclusive: with 400 questions each figure carries roughly +/-2.5 points of noise, only two models were checked, and the scorer is not comparable with published leaderboard numbers. A looser budget is a real trade: accepting Qwen2.5-0.5B 4-bit at 15% costs about 3 points of accuracy here.

## Signed validation reports

Each `SUCCESS` and `NO_CONFIG_MET_BUDGET` result carries a `signed_at` timestamp and a `signature` block:

```json
"signed_at": "2026-11-24T06:00:13+00:00",
"signature": { "alg": "ed25519", "public_key": "<base64>", "value": "<base64>" }
```

The signature covers the whole report in canonical JSON (sorted keys, no whitespace), so changing any field invalidates it. A calling agent can check a report without trusting the transport:

```python
import requests
from verify_report import verify_report

pub = requests.get(f"{URL}/public-key").json()["public_key"]
ok = verify_report(report, pub)   # True only if unmodified and signed by this key
```

[`results/signed_report_tinyllama.json`](results/signed_report_tinyllama.json) is a real signed report you can verify offline. Run `verify_report(report, report["signature"]["public_key"])` to check it.

**What a signature proves, and what it does not**
- It proves the report was not changed after signing and was signed by the holder of the private key.
- It does not prove the measurements are correct. It vouches for what the service reported, not for the quality of the evaluation.
- `verify_report` takes the trusted public key as an argument and does not trust the key embedded in the report, because an attacker could sign a forged report with their own key. Verifying against the report's own key only proves it was not edited.
- The canonical form follows Python's `json.dumps(sort_keys=True, separators=(",", ":"))`. A verifier or proxy in another language that re-serialises `10.0` as `10` will make a valid signature fail, so verify the report exactly as received or reproduce Python's formatting.
- In the Colab setup the signing key is generated fresh each session (`/content/signing_key.pem`), so the key you trust changes every time. A real deployment would keep a fixed key and publish its public half somewhere verifiers already trust. Never commit the private key (`*.pem` is in `.gitignore`).

## Use it as an MCP tool

[`mcp_server.py`](mcp_server.py) is a small MCP server that forwards tool calls to the running service. It exposes four tools: `submit_quantize_job`, `get_quantize_job`, `quantize_model` (synchronous, for models already measured) and `service_health`. For new models an agent should submit a job and poll it.

Run it on your own machine, since MCP clients such as Claude Desktop launch local processes:

```bash
pip install "mcp[cli]<2" requests
export QUANTIZE_URL=https://YOUR-TUNNEL.trycloudflare.com
export QUANTIZE_API_KEY=YOUR_KEY
python mcp_server.py
```

Claude Desktop configuration:

```json
{
  "mcpServers": {
    "quantize": {
      "command": "python",
      "args": ["/full/path/to/mcp_server.py"],
      "env": {
        "QUANTIZE_URL": "https://YOUR-TUNNEL.trycloudflare.com",
        "QUANTIZE_API_KEY": "YOUR_KEY"
      }
    }
  }
}
```

Pin `mcp<2`. Version 2 renamed `FastMCP`, and this code targets 1.x. The URL and key change every time the Colab service restarts.

## Run it outside Colab (Docker)

A `Dockerfile`, `docker-compose.yml` and `serve.py` run the same service as a container with a fixed API key (`API_KEY` environment variable), a persistent signing key and a persistent measurement cache. Requires a machine with an NVIDIA GPU and the NVIDIA Container Toolkit.

```bash
echo "API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" > .env
docker compose up -d --build
curl http://127.0.0.1:8000/health
```

See [`DEPLOY.md`](DEPLOY.md) for configuration, safe public access (stable tunnel or reverse proxy), what persists, and troubleshooting. **Status:** the container files have been written and syntax-checked but not yet built on a GPU machine.

## Tests

The suite needs no GPU, no network and no model downloads. The heavy libraries are mocked, and the tests use a temporary signing key and measurement cache, so they never touch your real ones.

```bash
pip install -r requirements-dev.txt
python -m pytest -q        # or: python -m unittest discover -s tests -t .
```

What is covered:
- **Planner:** which models fit, rejection reasons, inclusive boundaries, sort order, and that it never modifies the cache.
- **Quantizer logic** (with fake models): fallback from 4-bit to int8, stopping at the first passing format, `NO_CONFIG_MET_BUDGET`, the `bits` option, measurement reuse, and that the planner and quantizer always agree on the chosen format.
- **Signing:** valid reports verify; edits, moved signatures, malformed signatures and reports signed by another key are all rejected.
- **API-key gate:** every protected route needs the key and every public route stays open.
- **Fetcher:** remote code, pickle-only, already-quantized, gated and oversized models are refused, and fetch errors become clean refusals.
- **Endpoints and jobs:** refusals, signed results, per-request folders, job success and failure, unknown job IDs.
- **Repository checks** (`tests/test_repo.py`): the README's results table and the demo claim match `memory.json`, README links and Dockerfile `COPY` targets exist, `.dockerignore` and `*.pem` in `.gitignore` are present, no secrets or private keys are committed, notebooks carry no widget state, `serve.py` refuses a missing or short API key, and the signature verifier returns `False` (never raises) on malformed input.
- **HTTP wiring** (`tests/test_http.py`): the same routes through FastAPI. These tests skip themselves when `fastapi` and `httpx` are not installed.

These tests check the decision logic, not real quantization quality. Running actual models on a GPU is covered by the measured results above, not by the test suite.

## Repository layout

```
quantize_service.py        Fetcher, quantizer, validator, memory, signing and API server
verify_report.py           Standalone verifier for signed reports
tests/, requirements-dev.txt   Automated tests (no GPU needed), see Tests above
budget_planner.py          Cache-based planner behind POST /plan (pure Python)
tests/                     Test suite (no GPU or network needed)
requirements-dev.txt       Dependencies for running the tests
demo_agent.py              Example agent: plan, request, verify receipt, decide
serve.py                   Production entrypoint (env-var API key, no Colab)
Dockerfile, docker-compose.yml, .dockerignore   Container setup (see DEPLOY.md)
DEPLOY.md                  Deployment and operations notes
mcp_server.py              MCP wrapper around the HTTP service
memory.json                Measured results per model (the cache)
requirements.txt           Python dependencies
results/                   Raw JSON reports from the runs above
quantize_as_a_tool.ipynb   Development notebook (build log)
```

## Security notes

- Model files are read as safetensors. Pickle-only repos and `trust_remote_code` models are refused, since loading them means running untrusted content.
- The fetcher checks metadata first, so oversized or gated requests cost almost nothing.
- The tunnel URL is public. The API key is generated at startup, stored only in `/content/api_key.txt` and excluded by `.gitignore`. Never commit it.
- Quick tunnels have no rate limiting. Keep the URL private, and stop the runtime when you are done.

## Known limitations

- **Tunnel timeout (mitigated).** Cloudflare quick tunnels drop requests running longer than about 100 seconds. Use `POST /jobs` and poll for any model that is not yet in `memory.json`. The synchronous `/quantize` endpoint is only safe for models already measured. Jobs are held in memory, so a service restart forgets job IDs, although measured results persist in `memory.json`.
- **Perplexity is a proxy.** A small ARC-Easy check (see above) agrees with it in direction, but covers two models only. It is reliable for comparing formats of the same model, but the absolute values are not comparable with published WikiText-2 numbers because of the shortened evaluation. Task-level benchmarks are the next step.
- **Gated models untested.** The fetcher refuses gated repos by default. Running Gemma or Llama requires accepting the license (and sometimes waiting for manual approval) on the Hugging Face account, plus a read token stored as a secret, never in code or in the repository.
- **No artifact download yet.** A job leaves the compressed model on the server (`artifact_dir`), and the signed receipt carries its SHA-256, but the service has no endpoint to fetch it. The demo ends at a verified decision, not a deployment.
- **Small sample.** Six models, all at or below 3B parameters, on one T4.
- **bitsandbytes formats only.** NF4 and int8 reduce memory but do not speed up generation. Faster formats (GPTQ, AWQ, GGUF) are not included yet.
- **Size accounting.** The Qwen 4-bit figure (about 0.47 GB) was measured with a different method from the other rows. The like-for-like figure from the same method as the rest is 0.45 GB for plain NF4.
- **Colab is temporary.** The URL, key, downloaded models and artifacts disappear when the runtime resets. `memory.json` in this repository is what carries the measurements over.
- **Deterministic by design.** The three agents are plain Python classes, so the pipeline is predictable and testable. There is no LLM in the loop.

## Roadmap

- [x] Fetcher, quantizer and validator with fallback ladder
- [x] Per-model memory
- [x] HTTP API with key auth, GPU lock and size cap
- [x] MCP server
- [x] Planning endpoint and demo agent
- [x] Automated tests for the decision logic (the real-model path is covered by the measured results, not the suite)
- [x] Automated test suite (no GPU needed)
- [ ] Artifact download endpoint, and a `plan_models` MCP tool
- [x] Asynchronous jobs (submit, then poll)
- [x] Six models across five families tested (Qwen, Llama-style TinyLlama, Phi, SmolLM2, OLMo)
- [ ] Gated models (Gemma, Llama). The pipeline supports them when the Hugging Face account has been granted access and a token is provided, but access to Gemma was not granted during testing, so no results are claimed
- [x] Task benchmark spot-check (ARC-Easy, 2 models)
- [ ] Task benchmarks built into the service and run on larger samples
- [ ] GPTQ / AWQ / GGUF formats
- [x] Signed validation reports (Ed25519)
- [x] Dockerfile and deploy notes (written, not yet tested on a GPU host)
- [ ] A running, stable deployment with a fixed URL and a published signing key

## Tech stack

Python, PyTorch, Hugging Face `transformers` and `huggingface_hub`, bitsandbytes, FastAPI, Uvicorn, Cloudflare quick tunnels, Model Context Protocol (MCP).

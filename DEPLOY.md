# Deploying Quantize-as-a-Tool outside Colab

This runs the same service as the Colab notebook, but as a container with a fixed API key, a persistent signing key and a persistent measurement cache.

> **Status:** the files here were written and syntax-checked, but the image has **not been built or run on a GPU machine yet**. Treat the first build as a test. Version mismatches between PyTorch, `transformers` and `bitsandbytes` are the most likely thing to need adjusting (see "If the build fails").

## Requirements

- A machine with an **NVIDIA GPU** (the notebook was developed on a 16 GB T4) and a recent driver
- Docker, plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) so containers can see the GPU
- No GPU means no service: `bitsandbytes` quantization and our evaluation both run on CUDA

## Run it

```bash
git clone https://github.com/SUFIYAN2004/quantize-as-a-tool
cd quantize-as-a-tool

# choose a long random key and keep it out of git
echo "API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" > .env

docker compose up -d --build
docker compose logs -f          # first start prints the signing public key
curl http://127.0.0.1:8000/health
```

Call it exactly like the Colab version, with your key:

```bash
curl -X POST http://127.0.0.1:8000/jobs \
  -H "X-API-Key: $(grep API_KEY .env | cut -d= -f2)" -H "Content-Type: application/json" \
  -d '{"model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "accuracy_budget_pct": 10}'
```

Without Compose:

```bash
docker build -t quantize-as-a-tool .
docker run -d --gpus all -p 127.0.0.1:8000:8000 \
  -e API_KEY="your-long-random-key" \
  -v quantize-data:/content --name quantize quantize-as-a-tool
```

## Configuration

| Setting | Where | Meaning |
|---|---|---|
| `API_KEY` | environment variable (required, 16+ characters) | Value callers send in `X-API-Key`. The service refuses to start without it. |
| `HOST`, `PORT` | environment variable (optional) | Bind address (default `0.0.0.0`) and port (default `8000`) |
| `/content` | volume | Holds `memory.json`, `signing_key.pem`, the Hugging Face cache and `artifacts/` |

`/content` is used because the service code stores state there. It is just a directory inside the container, not Colab.

## What persists, and why it matters

- **`signing_key.pem` (on the volume).** The signing key is created on first start and reused after that, so the public key stays stable across restarts. This is what makes signed reports meaningful outside a single session. Back the volume up, and never commit the key.
- **Publish the public key.** The startup log and `GET /public-key` show it. Put it somewhere callers already trust, such as this repository's README, then have them verify with `verify_report(report, trusted_key)`.
- **`memory.json`.** The first start copies the repository's cache onto the volume, so the measured models are served instantly. New measurements are saved there.
- **Job table.** Jobs live in memory, so restarting the container forgets job IDs. Finished measurements are kept.

## Making it reachable safely

Compose binds the port to `127.0.0.1` on purpose. Do not publish port 8000 to the internet as is: it speaks plain HTTP, and a stolen key lets anyone spend your GPU.

- **Cloudflare named tunnel (free, needs a domain on Cloudflare):** gives a stable HTTPS URL with no open inbound port. Unlike the quick tunnels used in the notebook, the URL does not change. Note that Cloudflare still limits long requests, so callers should use `POST /jobs` and poll.
- **Reverse proxy (for example Caddy or nginx) on a server you control:** terminates HTTPS and forwards to `127.0.0.1:8000`.
- Either way, add rate limiting at the proxy, since the service itself has none.

Then point the MCP server at the stable URL:

```bash
QUANTIZE_URL=https://quantize.example.com QUANTIZE_API_KEY=... python mcp_server.py
```

## Where to run it

- **A machine you own with an NVIDIA GPU:** free apart from power, and the simplest way to test this.
- **A rented GPU server (RunPod, Lambda, Vast.ai, a cloud VM and similar):** works with the same commands. Prices and availability change, so check current rates, and stop the instance when you are not using it.
- **Google Colab:** cannot run Docker. Keep using the notebook there.

## Operating notes

- Run **one** container and one worker. The GPU lock and the job table are per process.
- Large models are refused above 3B parameters, which keeps requests inside a 16 GB GPU.
- Each request writes its artifact to `/content/artifacts/`. Delete old ones when disk fills up.
- `GET /docs`, `GET /health` and `GET /public-key` are open. `/quantize` and `/jobs` require the key.

## If the build fails

1. The base image pins PyTorch 2.5.1 with CUDA 12.4. If `pip install -r requirements.txt` tries to replace PyTorch, or the service fails to import, align the versions with the notebook environment. In Colab, run `!pip freeze | grep -iE "^(torch|transformers|bitsandbytes|accelerate|huggingface-hub|safetensors|datasets|fastapi|uvicorn|pydantic|starlette|cryptography)=="` and copy the output into a `requirements-lock.txt`, then use that file in the `Dockerfile` instead of `requirements.txt`.
2. A CUDA or driver error at start means the host driver is older than the image expects. Update the driver or choose a base image tag that matches it.
3. `bitsandbytes` errors about CUDA libraries usually mean the container cannot see the GPU. Check `docker run --rm --gpus all pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime nvidia-smi`.

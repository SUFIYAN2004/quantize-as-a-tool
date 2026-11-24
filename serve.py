"""Production entrypoint (no Colab, no tunnel).

Reads the API key from the API_KEY environment variable, keeps all state under
/content (mount a volume there), and serves quantize_service.app on port 8000.
"""
import os
import shutil
import sys

API_KEY = os.environ.get("API_KEY", "")
if len(API_KEY) < 16:
    sys.exit("Set the API_KEY environment variable (at least 16 characters).")

# quantize_service.py writes artifacts under /content, so state lives there.
DATA = "/content"
os.makedirs(f"{DATA}/artifacts", exist_ok=True)
os.environ.setdefault("MEM_PATH", f"{DATA}/memory.json")
os.environ.setdefault("SIGNING_KEY_PATH", f"{DATA}/signing_key.pem")

# First start on a fresh volume: seed the measurement cache from the repo copy.
seed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")
if not os.path.exists(os.environ["MEM_PATH"]) and os.path.exists(seed):
    shutil.copy(seed, os.environ["MEM_PATH"])

import quantize_service as qs  # noqa: E402  (reads the env vars above on import)
import uvicorn  # noqa: E402

print("Signing public key (publish this so callers can verify reports):")
print(qs._PUB_B64, flush=True)

if __name__ == "__main__":
    # One worker only: the GPU lock and the job table live in this process.
    uvicorn.run(
        qs.KeyGate(qs.app, API_KEY),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )

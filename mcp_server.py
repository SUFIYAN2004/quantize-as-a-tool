import os
import requests
from mcp.server.fastmcp import FastMCP

URL = os.environ.get("QUANTIZE_URL", "").rstrip("/")
KEY = os.environ.get("QUANTIZE_API_KEY", "")

mcp = FastMCP("quantize-as-a-tool")


def _slim(d: dict) -> dict:
    keep = ["status", "model_id", "winner", "winner_stats", "fp16_size_gb",
            "budget_pct", "artifact_sha256", "artifact_dir", "flags"]
    out = {k: d[k] for k in keep if k in d}
    out["attempts"] = [
        {k: a[k] for k in ("config", "ppl_increase_pct", "size_gb", "verdict")}
        for a in d.get("attempts", [])
    ]
    return out


@mcp.tool()
def quantize_model(model_id: str, accuracy_budget_pct: float = 10.0, bits: str = "auto") -> dict:
    """Compress a Hugging Face model (up to 3B params) to 4-bit or 8-bit.

    Returns the smallest format whose WikiText-2 perplexity increase stays within
    accuracy_budget_pct, plus the measured size, quality change and a SHA-256 of
    the artifact. bits is "auto", "4" or "8". Gated, remote-code, pickle-only or
    oversized models are refused with flags explaining why.
    """
    if not URL or not KEY:
        return {"status": "ERROR", "detail": "Set QUANTIZE_URL and QUANTIZE_API_KEY."}
    try:
        r = requests.post(
            f"{URL}/quantize",
            json={"model_id": model_id, "accuracy_budget_pct": accuracy_budget_pct, "bits": bits},
            headers={"X-API-Key": KEY},
            timeout=900,
        )
    except requests.RequestException as e:
        return {"status": "ERROR", "detail": f"Service unreachable: {type(e).__name__}"}
    if r.status_code == 401:
        return {"status": "UNAUTHORIZED", "detail": "Wrong API key."}
    if r.status_code != 200:
        return {"status": "ERROR", "detail": f"HTTP {r.status_code}"}
    return _slim(r.json())


@mcp.tool()
def service_health() -> dict:
    """Check whether the quantization service is reachable."""
    try:
        return requests.get(f"{URL}/health", timeout=20).json()
    except Exception as e:
        return {"ok": False, "detail": type(e).__name__}


if __name__ == "__main__":
    mcp.run()

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


def _call(method, path, **kw):
    if not URL or not KEY:
        return None, {"status": "ERROR", "detail": "Set QUANTIZE_URL and QUANTIZE_API_KEY."}
    try:
        r = requests.request(method, URL + path, headers={"X-API-Key": KEY}, timeout=120, **kw)
    except requests.RequestException as e:
        return None, {"status": "ERROR", "detail": "Service unreachable: " + type(e).__name__}
    if r.status_code == 401:
        return None, {"status": "UNAUTHORIZED", "detail": "Wrong API key."}
    if r.status_code == 404:
        return None, {"status": "NOT_FOUND", "detail": "Unknown job id (the service may have restarted)."}
    if r.status_code != 200:
        return None, {"status": "ERROR", "detail": "HTTP " + str(r.status_code)}
    return r.json(), None


@mcp.tool()
def submit_quantize_job(model_id: str, accuracy_budget_pct: float = 10.0, bits: str = "auto") -> dict:
    """Start compressing a Hugging Face model (up to 3B params) to 4-bit or 8-bit.

    Returns a job_id immediately. The work can take several minutes for a new
    model, so call get_quantize_job(job_id) every 10-30 seconds until the status
    is DONE or FAILED. accuracy_budget_pct is the largest perplexity increase
    you accept. bits is "auto", "4" or "8".
    """
    data, err = _call("POST", "/jobs", json={"model_id": model_id,
                      "accuracy_budget_pct": accuracy_budget_pct, "bits": bits})
    return err or data


@mcp.tool()
def get_quantize_job(job_id: str) -> dict:
    """Check a job. Status is QUEUED, RUNNING, DONE or FAILED. When DONE, the
    result holds the chosen format, measured size, perplexity change and the
    artifact SHA-256, or a REFUSED status with flags explaining why."""
    data, err = _call("GET", "/jobs/" + job_id)
    if err:
        return err
    out = {"job_id": job_id, "status": data.get("status"), "model_id": data.get("model_id")}
    if data.get("result"):
        out["result"] = _slim(data["result"])
    if data.get("error"):
        out["error"] = data["error"]
    return out


@mcp.tool()
def quantize_model(model_id: str, accuracy_budget_pct: float = 10.0, bits: str = "auto") -> dict:
    """Synchronous version. Only use for models already measured (fast). For new
    models use submit_quantize_job, because long requests can time out."""
    data, err = _call("POST", "/quantize", json={"model_id": model_id,
                      "accuracy_budget_pct": accuracy_budget_pct, "bits": bits})
    return err or _slim(data)


@mcp.tool()
def service_health() -> dict:
    """Check whether the quantization service is reachable."""
    data, err = _call("GET", "/health")
    return err or data


if __name__ == "__main__":
    mcp.run()

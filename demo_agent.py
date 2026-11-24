"""Demo: an agent picks a model for a constrained device and checks signed receipts.

Scenario: deploy a chat model to a device with a 1 GB memory budget for weights,
accepting at most 8% perplexity loss. The agent
  1. asks the service what fits (cheap, from the measurement cache),
  2. checks the plan's signature,
  3. requests the chosen model as a job and polls it,
  4. checks the receipt (signature, budget, size, model) before deciding,
  5. shows that a tampered receipt is rejected.

Settings (environment variables, all optional):
  QUANTIZE_URL         default http://127.0.0.1:8000
  QUANTIZE_API_KEY     default: read from /content/api_key.txt
  TRUSTED_PUBLIC_KEY   pin the signing key in advance (recommended)
  BUDGET_PCT, MAX_SIZE_GB   default 8 and 1.0
"""
import copy
import os
import sys
import time

import requests
from verify_report import verify_report

BASE = os.environ.get("QUANTIZE_URL", "http://127.0.0.1:8000").rstrip("/")
KEY = os.environ.get("QUANTIZE_API_KEY") or open("/content/api_key.txt").read().strip()
BUDGET = float(os.environ.get("BUDGET_PCT", 8.0))
MAX_GB = float(os.environ.get("MAX_SIZE_GB", 1.0))
H = {"X-API-Key": KEY}


def say(msg=""):
    print(msg, flush=True)


def stop(msg):
    say(f"\nSTOP: {msg}")
    sys.exit(1)


say(f"TASK: deploy a chat model to a device with {MAX_GB} GB for weights, "
    f"at most {BUDGET}% perplexity loss.\n")

# --- trust anchor ---
trusted = os.environ.get("TRUSTED_PUBLIC_KEY")
if not trusted:
    trusted = requests.get(f"{BASE}/public-key", timeout=30).json()["public_key"]
    say("(demo only: trusting the key the service just advertised. "
        "A real agent pins the key in advance.)\n")

# --- 1. plan ---
r = requests.post(f"{BASE}/plan", headers=H, timeout=60,
                  json={"accuracy_budget_pct": BUDGET, "max_size_gb": MAX_GB})
if r.status_code != 200:
    stop(f"/plan returned HTTP {r.status_code}")
plan = r.json()
if not verify_report(plan, trusted):
    stop("the plan's signature is invalid, refusing to act on it")
say("1. Plan received and signature verified.\n")

say("   Fits the limits (best quality first):")
for f in plan["fits"]:
    say(f"     {f['model_id']:<38} {f['config']:<17} +{f['ppl_increase_pct']}%  {f['size_gb']} GB")
say("\n   Rejected:")
for x in plan["rejected"]:
    say(f"     {x['model_id']:<38} {x['reason']:<12} {x['detail']}")

pick = plan["recommended"]
if not pick:
    stop("no known model meets both limits")
say(f"\n2. Choice: {pick['model_id']} as {pick['config']}\n")

# --- 3. job ---
job = requests.post(f"{BASE}/jobs", headers=H, timeout=60,
                    json={"model_id": pick["model_id"], "accuracy_budget_pct": BUDGET}).json()
say(f"3. Job submitted: {job.get('job_id')}")
last, j = None, {}
for _ in range(180):  # up to ~30 minutes
    j = requests.get(f"{BASE}/jobs/{job['job_id']}", headers=H, timeout=60).json()
    if j.get("status") != last:
        say(f"   {time.strftime('%H:%M:%S')} {j.get('status')}")
        last = j.get("status")
    if j.get("status") in ("DONE", "FAILED"):
        break
    time.sleep(10)
if j.get("status") != "DONE":
    stop(f"job did not finish: {j.get('status')} {j.get('error')}")
receipt = j["result"]

# --- 4. check the receipt before acting ---
stats = receipt.get("winner_stats") or {}
checks = {
    "status is SUCCESS": receipt.get("status") == "SUCCESS",
    "signature valid": verify_report(receipt, trusted),
    "same model as requested": receipt.get("model_id") == pick["model_id"],
    f"quality loss within {BUDGET}%": stats.get("ppl_increase_pct", 1e9) <= BUDGET,
    f"size within {MAX_GB} GB": stats.get("size_gb", 1e9) <= MAX_GB,
    "artifact hash present": bool(receipt.get("artifact_sha256")),
}
say("\n4. Receipt checks:")
for name, ok in checks.items():
    say(f"     [{'x' if ok else ' '}] {name}")
if not all(checks.values()):
    stop("receipt failed a check, not deploying")
say(f"\nDECISION: deploy {receipt['model_id']} ({receipt['winner']}, "
    f"+{stats['ppl_increase_pct']}%, {stats['size_gb']} GB)")
say(f"          artifact sha256 {receipt['artifact_sha256'][:16]}...")

# --- 5. tamper check ---
forged = copy.deepcopy(receipt)
forged["winner_stats"]["size_gb"] = 0.3
say(f"\n5. Forged receipt (size edited to 0.3 GB) accepted? "
    f"{verify_report(forged, trusted)}   <- must be False")

import os, json, time, math, gc, shutil, hashlib, secrets, threading, warnings
warnings.filterwarnings("ignore")
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MEM = os.environ.get("MEM_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json"))
MAX_PARAMS_M = 3000
api = HfApi()

# ---------------- helpers ----------------
def free_gpu():
    gc.collect(); torch.cuda.empty_cache()

def dir_size_gb(path):
    return round(sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(path) for f in fs) / 1e9, 3)

def folder_sha256(path):
    h = hashlib.sha256()
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with open(os.path.join(path, f), "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
    return h.hexdigest()

_WIKI = None
def eval_perplexity(model, tok, max_tokens=40000, seq_len=512):
    global _WIKI
    if _WIKI is None:
        from datasets import load_dataset
        d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        _WIKI = "\n\n".join(d["text"])
    ids = tok(_WIKI, return_tensors="pt").input_ids[:, :max_tokens]
    nll, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, ids.size(1) - 1, seq_len):
            chunk = ids[:, i:i + seq_len].to("cuda")
            if chunk.size(1) < 2:
                break
            out = model(chunk, labels=chunk)
            n = chunk.size(1) - 1
            nll += out.loss.item() * n
            count += n
    return math.exp(nll / count)

# ---------------- agent 1: fetcher ----------------
def _fetch_raw(model_id):
    info = api.model_info(model_id, files_metadata=True)
    files = {s.rfilename: (s.size or 0) for s in info.siblings}
    st = {n: s for n, s in files.items() if n.endswith(".safetensors")}
    pk = {n: s for n, s in files.items() if n.endswith((".bin", ".pt", ".pth", ".pkl"))}
    cfg = json.load(open(hf_hub_download(model_id, "config.json")))
    params = None
    if getattr(info, "safetensors", None) and info.safetensors.total:
        params = info.safetensors.total
    elif st:
        params = int(sum(st.values()) / 2)
    flags = []
    if cfg.get("auto_map"): flags.append("REMOTE_CODE: needs trust_remote_code")
    if pk and not st: flags.append("PICKLE_ONLY: unsafe weights")
    if getattr(info, "gated", False): flags.append("GATED: needs license and token")
    if cfg.get("quantization_config"): flags.append("ALREADY_QUANTIZED")
    return {
        "model_id": model_id,
        "architecture": (cfg.get("architectures") or ["?"])[0],
        "params_millions": round(params / 1e6, 1) if params else None,
        "flags": flags,
        "safe_to_proceed": not any(x.startswith(("REMOTE_CODE", "PICKLE_ONLY", "GATED")) for x in flags),
    }

def fetch_model_report(model_id):
    try:
        rep = _fetch_raw(model_id)
    except Exception as e:
        n = type(e).__name__
        flag = ("GATED: needs license acceptance and an HF token" if n == "GatedRepoError"
                else "NOT_FOUND: repo does not exist or is private" if n in ("RepositoryNotFoundError", "HfHubHTTPError")
                else f"FETCH_ERROR: {n}")
        return {"model_id": model_id, "params_millions": None, "flags": [flag], "safe_to_proceed": False}
    if rep["params_millions"] and rep["params_millions"] > MAX_PARAMS_M:
        rep["flags"].append(f"TOO_LARGE: {rep['params_millions']}M params > {MAX_PARAMS_M}M limit")
        rep["safe_to_proceed"] = False
    return rep

# ---------------- agent 2: quantizer (with memory) ----------------
def build_ladder():
    return [
        ("nf4+double_quant", BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)),
        ("nf4", BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype=torch.float16)),
        ("int8", BitsAndBytesConfig(load_in_8bit=True)),
    ]

def mem_load():
    return json.load(open(MEM)) if os.path.exists(MEM) else {}

def mem_save(m):
    os.makedirs(os.path.dirname(MEM), exist_ok=True)
    json.dump(m, open(MEM, "w"), indent=2)

def quantize_smart(model_id, budget=10.0, bits="auto", out_dir="/content/out_model"):
    mem = mem_load()
    entry = mem.setdefault(model_id, {"configs": {}})
    tok = AutoTokenizer.from_pretrained(model_id)
    ladder = build_ladder()
    if bits in (4, 8):
        ladder = [(n, c) for n, c in ladder if n.startswith("nf4") == (bits == 4)]
    if "fp16_ppl" not in entry:
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cuda")
        entry["fp16_gb"] = round(base.get_memory_footprint() / 1e9, 3)
        entry["fp16_ppl"] = eval_perplexity(base, tok)
        del base; free_gpu(); mem_save(mem)
    attempts, winner = [], None
    for name, cfg in ladder:
        known = entry["configs"].get(name)
        if known and known["ppl_increase_pct"] > budget:
            attempts.append({"config": name, **known, "source": "memory", "verdict": "FAIL"})
            continue
        qm = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=cfg, device_map="cuda")
        if known:
            res, src = dict(known), "memory"
        else:
            q_ppl = eval_perplexity(qm, tok)
            res = {"ppl_increase_pct": round((q_ppl - entry["fp16_ppl"]) / entry["fp16_ppl"] * 100, 2),
                   "size_gb": round(qm.get_memory_footprint() / 1e9, 3)}
            entry["configs"][name] = res; mem_save(mem); src = "measured"
        ok = res["ppl_increase_pct"] <= budget
        attempts.append({"config": name, **res, "source": src, "verdict": "PASS" if ok else "FAIL"})
        print(f"[{name}] +{res['ppl_increase_pct']}% size={res['size_gb']} GB -> {'PASS' if ok else 'FAIL'} ({src})")
        if ok:
            if os.path.exists(out_dir): shutil.rmtree(out_dir)
            qm.save_pretrained(out_dir); tok.save_pretrained(out_dir)
            winner = name; del qm; free_gpu(); break
        del qm; free_gpu()
    return {
        "model_id": model_id, "budget_pct": budget, "bits_requested": bits,
        "fp16_size_gb": entry["fp16_gb"], "winner": winner,
        "winner_stats": entry["configs"].get(winner), "attempts": attempts,
        "artifact_dir": out_dir if winner else None,
        "artifact_sha256": folder_sha256(out_dir) if winner else None,
        "status": "SUCCESS" if winner else "NO_CONFIG_MET_BUDGET",
    }

# ---------------- agent 3: validator ----------------
def verify_artifact(artifact_dir, expected_sha256):
    ok = os.path.exists(artifact_dir) and folder_sha256(artifact_dir) == expected_sha256
    return {"hash_matches": ok, "size_gb": dir_size_gb(artifact_dir) if ok else None}

# ---------------- service ----------------
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse
from budget_planner import plan_from_memory

app = FastAPI(title="Quantize-as-a-tool")
gpu_lock = threading.Lock()

class Req(BaseModel):
    model_id: str
    accuracy_budget_pct: float = 10.0
    bits: str = "auto"

@app.post("/quantize")
def quantize_endpoint(req: Req):
    insp = fetch_model_report(req.model_id)
    if not insp["safe_to_proceed"]:
        return {"status": "REFUSED", "flags": insp["flags"]}
    bits = int(req.bits) if req.bits in ("4", "8") else "auto"
    out_dir = f"/content/artifacts/{req.model_id.replace('/', '_')}_{req.accuracy_budget_pct}"
    with gpu_lock:
        return sign_report(quantize_smart(req.model_id, req.accuracy_budget_pct, bits, out_dir))

@app.get("/health")
def health():
    return {"ok": True}

class PlanReq(BaseModel):
    accuracy_budget_pct: float = 10.0
    max_size_gb: float | None = None
    model_ids: list[str] | None = None

@app.post("/plan")
def plan_endpoint(req: PlanReq):
    """Cheap, GPU-free: which already-measured models fit these limits?"""
    return sign_report(plan_from_memory(mem_load(), req.accuracy_budget_pct,
                                        req.max_size_gb, req.model_ids))

class KeyGate:
    def __init__(self, inner, key):
        self.inner, self.key = inner, key.encode()
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and (scope["path"] in ("/quantize", "/plan") or scope["path"].startswith("/jobs")):
            supplied = dict(scope["headers"]).get(b"x-api-key", b"")
            if not secrets.compare_digest(supplied, self.key):
                await JSONResponse({"status": "UNAUTHORIZED"}, status_code=401)(scope, receive, send)
                return
        await self.inner(scope, receive, send)


import uuid
JOBS = {}

def _run_job(job_id, req):
    JOBS[job_id]["status"] = "RUNNING"
    try:
        JOBS[job_id]["result"] = quantize_endpoint(req)
        JOBS[job_id]["status"] = "DONE"
    except Exception as e:
        JOBS[job_id]["status"] = "FAILED"
        JOBS[job_id]["error"] = type(e).__name__

@app.post("/jobs")
def submit_job(req: Req):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "QUEUED", "model_id": req.model_id}
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id, "status": "QUEUED"}

@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    j = JOBS.get(job_id)
    return j if j else JSONResponse({"status": "NOT_FOUND"}, status_code=404)


from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import base64, datetime

KEY_PATH = os.environ.get("SIGNING_KEY_PATH", "/content/signing_key.pem")

def _load_signing_key():
    if os.path.exists(KEY_PATH):
        return serialization.load_pem_private_key(open(KEY_PATH, "rb").read(), password=None)
    k = Ed25519PrivateKey.generate()
    open(KEY_PATH, "wb").write(k.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    return k

_SIGN_KEY = _load_signing_key()
_PUB_B64 = base64.b64encode(_SIGN_KEY.public_key().public_bytes(
    serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()

def _canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()

def sign_report(result):
    if result.get("status") not in ("SUCCESS", "NO_CONFIG_MET_BUDGET", "PLAN"):
        return result
    body = {k: v for k, v in result.items() if k != "signature"}
    body["signed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    sig = _SIGN_KEY.sign(_canon(body))
    body["signature"] = {"alg": "ed25519", "public_key": _PUB_B64, "value": base64.b64encode(sig).decode()}
    return body

@app.get("/public-key")
def public_key():
    return {"alg": "ed25519", "public_key": _PUB_B64}

def start_service(port=8000, tunnel=True):
    import uvicorn
    key = secrets.token_urlsafe(24)
    open("/content/api_key.txt", "w").write(key)
    server = uvicorn.Server(uvicorn.Config(KeyGate(app, key), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(4)
    url = None
    if tunnel:
        from pycloudflared import try_cloudflare
        url = try_cloudflare(port=port).tunnel
        open("/content/public_url.txt", "w").write(url)
    return server, key, url

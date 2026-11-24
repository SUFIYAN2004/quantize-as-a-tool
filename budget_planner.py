"""Answers "which known models fit my limits?" from the measurement cache.

Pure Python, no GPU: an agent can call this cheaply before it asks for an
expensive quantization job. It mirrors the quantizer's rule, so the config it
recommends for a model is the one a job with the same budget will deliver.
"""

# Same order the quantizer tries them (smallest first).
LADDER = ["nf4+double_quant", "nf4", "int8"]

SIZE_BASIS = "weights memory footprint in GB; excludes activations and KV cache"


def plan_from_memory(mem, budget_pct, max_size_gb=None, model_ids=None):
    ids = list(model_ids) if model_ids else sorted(mem)
    fits, rejected = [], []

    for mid in ids:
        entry = mem.get(mid)
        cfgs = (entry or {}).get("configs") or {}
        if not cfgs:
            rejected.append({"model_id": mid, "reason": "not_measured",
                             "detail": "no measurements yet; submit a quantize job to measure it"})
            continue

        unmeasured = [n for n in LADDER if n not in cfgs]
        winner = next((n for n in LADDER
                       if n in cfgs and cfgs[n]["ppl_increase_pct"] <= budget_pct), None)

        if winner is None:
            best = min(c["ppl_increase_pct"] for c in cfgs.values())
            detail = f"best measured perplexity increase is {best}%, above the {budget_pct}% budget"
            if unmeasured:
                detail += f"; not yet measured: {', '.join(unmeasured)}"
            rejected.append({"model_id": mid, "reason": "over_budget", "detail": detail})
            continue

        stats = cfgs[winner]
        item = {"model_id": mid, "config": winner,
                "ppl_increase_pct": stats["ppl_increase_pct"],
                "size_gb": stats["size_gb"],
                "fp16_size_gb": (entry or {}).get("fp16_gb")}

        if max_size_gb is not None and stats["size_gb"] > max_size_gb:
            rejected.append({"model_id": mid, "reason": "too_big",
                             "detail": f"{winner} needs {stats['size_gb']} GB, limit is {max_size_gb} GB"})
            continue
        fits.append(item)

    fits.sort(key=lambda f: (f["ppl_increase_pct"], f["size_gb"]))
    return {
        "status": "PLAN",
        "source": "measurement_cache",
        "budget_pct": budget_pct,
        "max_size_gb": max_size_gb,
        "size_basis": SIZE_BASIS,
        "recommended": fits[0] if fits else None,
        "fits": fits,
        "rejected": rejected,
    }

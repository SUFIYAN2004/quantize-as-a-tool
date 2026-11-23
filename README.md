# Quantize-as-a-Tool

Model compression that AI agents can call over HTTP, with an accuracy budget and a verification report.

An agent sends `{model_id, accuracy_budget_pct}` and gets back the smallest 4-bit or 8-bit
version of a Hugging Face model that stays within its perplexity budget.

## How it works
1. **Fetcher** reads metadata only (no weights) and refuses gated, pickle-only, remote-code or oversized (>3B) models.
2. **Quantizer** tries NF4+double-quant, NF4, then int8, and stops at the first one that meets the budget.
3. **Validator** compares WikiText-2 perplexity with the fp16 baseline and records a SHA-256 of the artifact.

A per-model memory (`memory.json`) makes repeat requests instant.

## Measured results (Tesla T4)
| Model | Format chosen | Size vs fp16 | Perplexity change | Budget |
|---|---|---|---|---|
| Qwen2.5-0.5B | int8 | 0.63 GB vs 0.99 GB | +0.89% | 10% |
| Qwen2.5-0.5B | NF4 4-bit | 0.47 GB | +12.92% | 15% |
| TinyLlama-1.1B | NF4 4-bit | 0.75 GB vs 2.2 GB | +5.42% | 10% |

The right format depends on the model: 4-bit fails a 10% budget on Qwen 0.5B but passes on TinyLlama 1.1B.

## Run it (Google Colab, T4 GPU)
```python
!pip install -q -r requirements.txt
%run quantize_service.py
server, key, url = start_service()
print(url)   # API key is saved in /content/api_key.txt
```

## Call it
```bash
curl -X POST $URL/quantize \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "accuracy_budget_pct": 10}'
```

## Limits
Perplexity is a proxy for quality (task benchmarks are next). Tested on two small models. Deterministic Python agents, no LLM in the loop.
Roadmap: MCP wrapper, more architectures, GPTQ/AWQ/GGUF, signed reports.

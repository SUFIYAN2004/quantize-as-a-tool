"""A measurement cache shaped like memory.json, using the numbers measured in this project."""

SAMPLE_MEMORY = {
    "Qwen/Qwen2.5-0.5B": {"fp16_gb": 0.988, "configs": {
        "nf4+double_quant": {"ppl_increase_pct": 12.92, "size_gb": 0.466},
        "nf4": {"ppl_increase_pct": 12.95, "size_gb": 0.451},
        "int8": {"ppl_increase_pct": 0.89, "size_gb": 0.63}}},
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"fp16_gb": 2.2, "configs": {
        "nf4+double_quant": {"ppl_increase_pct": 5.42, "size_gb": 0.747}}},
    "microsoft/phi-2": {"configs": {
        "nf4+double_quant": {"ppl_increase_pct": 5.94, "size_gb": 1.784}}},
    "Qwen/Qwen2.5-1.5B": {"configs": {
        "nf4+double_quant": {"ppl_increase_pct": 8.66, "size_gb": 1.122}}},
    "HuggingFaceTB/SmolLM2-1.7B": {"fp16_gb": 3.423, "configs": {
        "nf4+double_quant": {"ppl_increase_pct": 15.29, "size_gb": 1.007},
        "nf4": {"ppl_increase_pct": 15.39, "size_gb": 1.007},
        "int8": {"ppl_increase_pct": 2.64, "size_gb": 1.812}}},
    "allenai/OLMo-1B-hf": {"configs": {
        "nf4+double_quant": {"ppl_increase_pct": 7.21, "size_gb": 0.949}}},
    "empty/model": {"configs": {}},
}

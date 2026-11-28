"""Tests the quantizer's decision logic with fake models: no GPU, no downloads."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import tests  # noqa: F401
import quantize_service as qs
from budget_planner import plan_from_memory

NAMES = ("nf4+double_quant", "nf4", "int8")
BASE_PPL, BASE_GB = 17.0, 0.988


class FakeModel:
    def __init__(self, cfg, size):
        self.cfg, self.size = cfg, size

    def get_memory_footprint(self):
        return int(self.size * 1e9)

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.safetensors"), "wb") as f:
            f.write(b"x")


class QuantizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.inc = {"nf4+double_quant": 12.92, "nf4": 12.95, "int8": 0.89}
        self.size = {"nf4+double_quant": 0.466, "nf4": 0.451, "int8": 0.63}
        self.loads, self.evals = [], []

        def load(model_id, **kw):
            cfg = kw.get("quantization_config")
            self.loads.append(cfg)
            return FakeModel(cfg, BASE_GB if cfg is None else self.size[cfg])

        def evaluate(model, tok):
            self.evals.append(model.cfg)
            return BASE_PPL if model.cfg is None else BASE_PPL * (1 + self.inc[model.cfg] / 100)

        patches = {
            "MEM": os.path.join(self.tmp, "memory.json"),
            "build_ladder": lambda: [(n, n) for n in NAMES],
            "AutoTokenizer": mock.MagicMock(),
            "AutoModelForCausalLM": SimpleNamespace(from_pretrained=load),
            "eval_perplexity": evaluate,
            "folder_sha256": lambda path: "deadbeef",
            "free_gpu": lambda: None,
        }
        for name, value in patches.items():
            p = mock.patch.object(qs, name, value)
            p.start()
            self.addCleanup(p.stop)

    def run_q(self, budget=10.0, bits="auto", name="out", model="org/m"):
        out_dir = os.path.join(self.tmp, name)
        return qs.quantize_smart(model, budget=budget, bits=bits, out_dir=out_dir), out_dir

    def forget(self):
        self.loads.clear()
        self.evals.clear()

    # ---- ladder behaviour ----
    def test_falls_back_to_int8_when_4bit_fails_the_budget(self):
        res, out_dir = self.run_q(10.0)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["winner"], "int8")
        self.assertEqual([a["verdict"] for a in res["attempts"]], ["FAIL", "FAIL", "PASS"])
        self.assertEqual({a["source"] for a in res["attempts"]}, {"measured"})
        self.assertEqual(self.loads, [None, "nf4+double_quant", "nf4", "int8"])
        self.assertEqual(res["winner_stats"], {"ppl_increase_pct": 0.89, "size_gb": 0.63})
        self.assertEqual(res["fp16_size_gb"], BASE_GB)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "model.safetensors")))
        self.assertEqual(res["artifact_sha256"], "deadbeef")

    def test_stops_at_the_first_format_that_passes(self):
        self.inc["nf4+double_quant"] = 5.0
        res, _ = self.run_q(10.0)
        self.assertEqual(res["winner"], "nf4+double_quant")
        self.assertEqual(self.loads, [None, "nf4+double_quant"])

    def test_no_format_meets_the_budget(self):
        res, out_dir = self.run_q(0.5)
        self.assertEqual(res["status"], "NO_CONFIG_MET_BUDGET")
        self.assertIsNone(res["winner"])
        self.assertIsNone(res["artifact_dir"])
        self.assertIsNone(res["artifact_sha256"])
        self.assertFalse(os.path.exists(out_dir))

    def test_bits_option_restricts_the_formats_tried(self):
        res, _ = self.run_q(10.0, bits=4)
        self.assertEqual(res["status"], "NO_CONFIG_MET_BUDGET")
        self.assertEqual(self.loads, [None, "nf4+double_quant", "nf4"])
        self.forget()
        res, _ = self.run_q(10.0, bits=8, model="org/other")
        self.assertEqual(res["winner"], "int8")
        self.assertEqual(self.loads, [None, "int8"])

    # ---- memory ----
    def test_repeat_request_reuses_measurements(self):
        self.run_q(10.0)
        self.forget()
        res, _ = self.run_q(10.0, name="out2")
        self.assertEqual({a["source"] for a in res["attempts"]}, {"memory"})
        self.assertEqual(self.evals, [])
        self.assertEqual(self.loads, ["int8"])   # only the winner is loaded, to save it
        self.assertEqual(res["winner"], "int8")

    def test_looser_budget_picks_the_smaller_known_format(self):
        self.run_q(10.0)
        self.forget()
        res, _ = self.run_q(15.0, name="out15")
        self.assertEqual(res["winner"], "nf4+double_quant")
        self.assertEqual(self.evals, [])
        self.assertEqual(self.loads, ["nf4+double_quant"])

    def test_baseline_is_measured_once_per_model(self):
        self.run_q(10.0)
        self.run_q(15.0, name="out15")
        self.assertEqual(self.loads.count(None), 1)
        self.assertEqual(self.evals.count(None), 1)

    def test_measurements_are_persisted(self):
        self.run_q(10.0)
        with open(qs.MEM) as f:
            saved = json.load(f)["org/m"]
        self.assertEqual(saved["fp16_gb"], BASE_GB)
        self.assertEqual(set(saved["configs"]), set(NAMES))
        self.assertEqual(saved["configs"]["int8"]["ppl_increase_pct"], 0.89)

    # ---- the planner promises what a job will deliver ----
    def test_planner_and_quantizer_agree(self):
        cache = {"org/m": {"fp16_ppl": BASE_PPL, "fp16_gb": BASE_GB, "configs": {
            n: {"ppl_increase_pct": self.inc[n], "size_gb": self.size[n]} for n in NAMES}}}
        with open(qs.MEM, "w") as f:
            json.dump(cache, f)
        for i, budget in enumerate((0.5, 0.89, 1.0, 12.92, 12.93, 13.0, 20.0)):
            plan = plan_from_memory(cache, budget)
            res, _ = self.run_q(budget, name=f"agree{i}")
            if plan["fits"]:
                self.assertEqual(res["winner"], plan["fits"][0]["config"], budget)
            else:
                self.assertIsNone(res["winner"], budget)
                self.assertEqual(plan["rejected"][0]["reason"], "over_budget", budget)


if __name__ == "__main__":
    unittest.main()

import unittest

import tests  # noqa: F401  (installs stubs, sets sys.path)
from budget_planner import plan_from_memory
from tests.sample_data import SAMPLE_MEMORY as MEM


def ids(items):
    return [i["model_id"] for i in items]


class PlannerTests(unittest.TestCase):
    def test_demo_scenario_8pct_1gb(self):
        p = plan_from_memory(MEM, 8.0, 1.0)
        self.assertEqual(ids(p["fits"]), [
            "Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "allenai/OLMo-1B-hf"])
        self.assertEqual(p["recommended"]["model_id"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(p["recommended"]["config"], "int8")
        self.assertEqual(p["status"], "PLAN")

    def test_rejection_reasons(self):
        p = plan_from_memory(MEM, 8.0, 1.0)
        why = {r["model_id"]: r["reason"] for r in p["rejected"]}
        self.assertEqual(why["microsoft/phi-2"], "too_big")
        self.assertEqual(why["HuggingFaceTB/SmolLM2-1.7B"], "too_big")
        self.assertEqual(why["Qwen/Qwen2.5-1.5B"], "over_budget")
        self.assertEqual(why["empty/model"], "not_measured")

    def test_over_budget_mentions_unmeasured_formats(self):
        p = plan_from_memory(MEM, 8.0, 1.0, ["Qwen/Qwen2.5-1.5B"])
        self.assertIn("nf4", p["rejected"][0]["detail"])
        self.assertIn("int8", p["rejected"][0]["detail"])

    def test_no_size_limit_lets_big_models_in(self):
        p = plan_from_memory(MEM, 8.0, None)
        self.assertEqual(len(p["fits"]), 5)

    def test_looser_budget_picks_smaller_format(self):
        tight = plan_from_memory(MEM, 10.0, None, ["Qwen/Qwen2.5-0.5B"])["fits"][0]
        loose = plan_from_memory(MEM, 15.0, None, ["Qwen/Qwen2.5-0.5B"])["fits"][0]
        self.assertEqual(tight["config"], "int8")
        self.assertEqual(loose["config"], "nf4+double_quant")
        self.assertLess(loose["size_gb"], tight["size_gb"])

    def test_boundaries_are_inclusive(self):
        at_budget = plan_from_memory(MEM, 5.42, 0.747, ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
        self.assertEqual(len(at_budget["fits"]), 1)
        over = plan_from_memory(MEM, 5.41, 0.747, ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
        self.assertEqual(over["rejected"][0]["reason"], "over_budget")
        small = plan_from_memory(MEM, 5.42, 0.746, ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
        self.assertEqual(small["rejected"][0]["reason"], "too_big")

    def test_nothing_fits(self):
        p = plan_from_memory(MEM, 0.1, 0.1)
        self.assertIsNone(p["recommended"])
        self.assertEqual(p["fits"], [])

    def test_empty_cache_and_unknown_model(self):
        self.assertIsNone(plan_from_memory({}, 10.0, 1.0)["recommended"])
        p = plan_from_memory(MEM, 10.0, 1.0, ["nobody/home"])
        self.assertEqual(p["rejected"][0]["reason"], "not_measured")

    def test_fits_sorted_by_quality_then_size(self):
        p = plan_from_memory(MEM, 20.0, None)
        keys = [(f["ppl_increase_pct"], f["size_gb"]) for f in p["fits"]]
        self.assertEqual(keys, sorted(keys))

    def test_uses_first_passing_format_in_quantizer_order(self):
        mem = {"m/x": {"configs": {
            "nf4+double_quant": {"ppl_increase_pct": 12.0, "size_gb": 0.4},
            "nf4": {"ppl_increase_pct": 6.0, "size_gb": 0.45},
            "int8": {"ppl_increase_pct": 1.0, "size_gb": 0.6}}}}
        self.assertEqual(plan_from_memory(mem, 8.0)["fits"][0]["config"], "nf4")
        self.assertEqual(plan_from_memory(mem, 2.0)["fits"][0]["config"], "int8")

    def test_does_not_modify_the_cache(self):
        import copy
        before = copy.deepcopy(MEM)
        plan_from_memory(MEM, 8.0, 1.0)
        self.assertEqual(MEM, before)


if __name__ == "__main__":
    unittest.main()

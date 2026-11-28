import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import tests  # noqa: F401
import quantize_service as qs


def make_info(files, params=None, gated=False):
    siblings = [SimpleNamespace(rfilename=name, size=size) for name, size in files.items()]
    st = SimpleNamespace(total=params) if params else None
    return SimpleNamespace(siblings=siblings, safetensors=st, gated=gated)


class FetcherTests(unittest.TestCase):
    def fetch(self, info, config=None):
        cfg = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}
        cfg.update(config or {})
        path = os.path.join(tempfile.mkdtemp(), "config.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        fake_api = SimpleNamespace(model_info=lambda model_id, files_metadata=True: info)
        with mock.patch.object(qs, "api", fake_api), \
             mock.patch.object(qs, "hf_hub_download", lambda model_id, name: path):
            return qs.fetch_model_report("org/model")

    def test_clean_model_is_safe(self):
        rep = self.fetch(make_info({"model.safetensors": 988_000_000}, params=494_000_000))
        self.assertTrue(rep["safe_to_proceed"])
        self.assertEqual(rep["flags"], [])
        self.assertEqual(rep["params_millions"], 494.0)
        self.assertEqual(rep["architecture"], "Qwen2ForCausalLM")

    def test_param_count_falls_back_to_file_size(self):
        rep = self.fetch(make_info({"model.safetensors": 2_000_000_000}))
        self.assertEqual(rep["params_millions"], 1000.0)

    def test_remote_code_is_blocked(self):
        rep = self.fetch(make_info({"model.safetensors": 1}, params=1_000_000),
                         {"auto_map": {"AutoModel": "modeling.Custom"}})
        self.assertFalse(rep["safe_to_proceed"])
        self.assertTrue(any(f.startswith("REMOTE_CODE") for f in rep["flags"]))

    def test_pickle_only_is_blocked(self):
        rep = self.fetch(make_info({"pytorch_model.bin": 1_000_000_000}))
        self.assertFalse(rep["safe_to_proceed"])
        self.assertTrue(any(f.startswith("PICKLE_ONLY") for f in rep["flags"]))

    def test_pickle_next_to_safetensors_is_allowed(self):
        rep = self.fetch(make_info({"pytorch_model.bin": 1, "model.safetensors": 2_000_000},
                                   params=1_000_000))
        self.assertTrue(rep["safe_to_proceed"])

    def test_gated_is_blocked(self):
        rep = self.fetch(make_info({"model.safetensors": 1}, params=1_000_000, gated=True))
        self.assertFalse(rep["safe_to_proceed"])
        self.assertTrue(any(f.startswith("GATED") for f in rep["flags"]))

    def test_already_quantized_is_blocked(self):
        rep = self.fetch(make_info({"model.safetensors": 1}, params=1_000_000),
                         {"quantization_config": {"quant_method": "gptq"}})
        self.assertFalse(rep["safe_to_proceed"])
        self.assertIn("ALREADY_QUANTIZED", rep["flags"])

    def test_size_cap(self):
        over = self.fetch(make_info({"model.safetensors": 1}, params=7_615_600_000))
        self.assertFalse(over["safe_to_proceed"])
        self.assertTrue(any(f.startswith("TOO_LARGE") for f in over["flags"]))
        at_limit = self.fetch(make_info({"model.safetensors": 1}, params=3_000_000_000))
        self.assertTrue(at_limit["safe_to_proceed"])

    def test_fetch_errors_become_refusals(self):
        def failing(name):
            err = type(name, (Exception,), {})
            def boom(model_id, files_metadata=True):
                raise err("simulated")
            return SimpleNamespace(model_info=boom)

        expected = {"GatedRepoError": "GATED", "RepositoryNotFoundError": "NOT_FOUND",
                    "SomethingElse": "FETCH_ERROR"}
        for name, prefix in expected.items():
            with mock.patch.object(qs, "api", failing(name)):
                rep = qs.fetch_model_report("org/model")
            self.assertFalse(rep["safe_to_proceed"], name)
            self.assertTrue(rep["flags"][0].startswith(prefix), (name, rep["flags"]))


if __name__ == "__main__":
    unittest.main()

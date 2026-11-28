"""Checks on the repository itself: that the README's numbers match the data, that
links and Docker files point at things that exist, that no secrets are committed,
and that the signature verifier fails closed on malformed input.

Pure Python: no GPU, no network, no FastAPI needed.
"""
import copy
import json
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from budget_planner import LADDER, plan_from_memory  # noqa: E402
from verify_report import verify_report  # noqa: E402


def p(*parts):
    return os.path.join(ROOT, *parts)


def load_json(*parts):
    with open(p(*parts)) as f:
        return json.load(f)


with open(p("README.md"), encoding="utf-8") as _f:
    README = _f.read()


class VerifierNeverRaises(unittest.TestCase):
    """A verifier must fail closed: malformed input means 'not valid', not a crash."""

    def setUp(self):
        self.report = load_json("results", "signed_report_tinyllama.json")
        self.pub = self.report["signature"]["public_key"]

    def test_sample_report_is_valid(self):
        self.assertTrue(verify_report(self.report, self.pub))

    def check(self, report, trusted=None):
        try:
            result = verify_report(report, trusted or self.pub)
        except Exception as e:  # noqa: BLE001
            self.fail(f"raised {type(e).__name__}: {e}")
        self.assertFalse(result)

    def test_not_a_dict(self):
        for bad in (None, [], "text", 42):
            self.check(bad)

    def test_missing_or_odd_signature(self):
        for sig in (None, "string", 7, [], {}):
            r = copy.deepcopy(self.report)
            r["signature"] = sig
            self.check(r)
        r = copy.deepcopy(self.report)
        del r["signature"]
        self.check(r)

    def test_bad_signature_value(self):
        for value in (None, 123, "not base64!!", "", "AAAA"):
            r = copy.deepcopy(self.report)
            r["signature"]["value"] = value
            self.check(r)

    def test_bad_trusted_key(self):
        r = copy.deepcopy(self.report)
        r["signature"]["public_key"] = "AAAA"
        self.check(r, trusted="AAAA")
        self.check(self.report, trusted=12345)

    def test_integer_float_difference_breaks_the_signature(self):
        # Documented limitation: the canonical form follows Python's json.dumps,
        # so re-serialising 10.0 as 10 (as some other languages do) invalidates it.
        r = copy.deepcopy(self.report)
        r["budget_pct"] = int(r["budget_pct"])
        self.assertFalse(verify_report(r, self.pub))


class ServeEntrypoint(unittest.TestCase):
    def run_serve(self, env_extra):
        env = {k: v for k, v in os.environ.items() if k != "API_KEY"}
        env.update(env_extra)
        return subprocess.run([sys.executable, p("serve.py")], env=env,
                              capture_output=True, text=True, timeout=60)

    def test_refuses_to_start_without_a_key(self):
        r = self.run_serve({})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("API_KEY", r.stderr)

    def test_refuses_a_short_key(self):
        r = self.run_serve({"API_KEY": "short"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("API_KEY", r.stderr)


class MeasurementCache(unittest.TestCase):
    def test_schema(self):
        mem = load_json("memory.json")
        self.assertGreaterEqual(len(mem), 6)
        for model, entry in mem.items():
            self.assertIn("/", model)
            self.assertTrue(entry["configs"], f"{model} has no configs")
            for name, stats in entry["configs"].items():
                self.assertIn(name, LADDER, f"{model}: unknown config {name}")
                self.assertIsInstance(stats["ppl_increase_pct"], (int, float))
                self.assertGreater(stats["size_gb"], 0)
            if "fp16_gb" in entry:
                self.assertGreater(entry["fp16_gb"], 0)

    def test_results_files_are_valid_json(self):
        names = [n for n in os.listdir(p("results")) if n.endswith(".json")]
        self.assertTrue(names)
        for n in names:
            load_json("results", n)


class ReadmeMatchesData(unittest.TestCase):
    NAMES = {"Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
             "TinyLlama-1.1B-Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
             "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B", "Phi-2": "microsoft/phi-2",
             "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B", "OLMo-1B": "allenai/OLMo-1B-hf"}
    FORMATS = {"int8": "int8", "NF4 4-bit": "nf4+double_quant"}

    def test_results_table_matches_memory_json(self):
        mem = load_json("memory.json")
        row = re.compile(r"^\|\s*([^|]+?)\s*\|\s*[\d.]+B\s*\|\s*(int8|NF4 4-bit)\s*\|"
                         r"\s*([\d.]+) GB[^|]*\|\s*\+([\d.]+)%\s*\|$", re.M)
        rows = row.findall(README)
        self.assertGreaterEqual(len(rows), 6, "could not find the results table")
        for name, fmt, size, ppl in rows:
            stats = mem[self.NAMES[name]]["configs"][self.FORMATS[fmt]]
            self.assertAlmostEqual(float(ppl), stats["ppl_increase_pct"], places=2, msg=name)
            self.assertAlmostEqual(float(size), stats["size_gb"], delta=0.006, msg=name)

    def test_demo_claim_holds(self):
        # README: at 8% and 1 GB, three models fit (Qwen2.5-0.5B int8, TinyLlama, OLMo),
        # Phi-2 and SmolLM2 are too big, Qwen2.5-1.5B is over budget.
        plan = plan_from_memory(load_json("memory.json"), 8.0, 1.0)
        self.assertEqual([f["model_id"] for f in plan["fits"]],
                         ["Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                          "allenai/OLMo-1B-hf"])
        self.assertEqual(plan["recommended"]["config"], "int8")
        reasons = {r["model_id"]: r["reason"] for r in plan["rejected"]}
        self.assertEqual(reasons["microsoft/phi-2"], "too_big")
        self.assertEqual(reasons["HuggingFaceTB/SmolLM2-1.7B"], "too_big")
        self.assertEqual(reasons["Qwen/Qwen2.5-1.5B"], "over_budget")

    def test_benchmark_table_matches_json(self):
        bench = load_json("results", "benchmarks_arc.json")["results"]
        self.assertEqual(bench["Qwen/Qwen2.5-0.5B"]["fp16"], 59.2)
        self.assertEqual(bench["Qwen/Qwen2.5-0.5B"]["nf4+double_quant"], 56.2)
        self.assertEqual(bench["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]["nf4+double_quant"], 54.8)
        for needle in ("59.2%", "56.2%", "55.2%", "54.8%"):
            self.assertIn(needle, README)


class RepositoryHygiene(unittest.TestCase):
    def test_readme_local_links_exist(self):
        broken = []
        for target in re.findall(r"\]\(([^)\s\"']+)\)", README):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if not os.path.exists(p(target.split("#")[0])):
                broken.append(target)
        self.assertEqual(broken, [], f"README links to missing files: {broken}")

    def test_dockerfile_copies_files_that_exist(self):
        with open(p("Dockerfile")) as f:
            for line in f:
                if line.startswith("COPY "):
                    sources = [t for t in line.split()[1:-1] if not t.startswith("--")]
                    for s in sources:
                        self.assertTrue(os.path.exists(p(s)), f"Dockerfile copies missing {s}")

    def test_dockerignore_has_the_leading_dot(self):
        self.assertTrue(os.path.exists(p(".dockerignore")),
                        "rename 'dockerignore' to '.dockerignore'")

    def test_gitignore_blocks_secrets(self):
        with open(p(".gitignore")) as f:
            ignored = f.read()
        for entry in ("api_key.txt", ".env", "*.safetensors", "*.pem"):
            self.assertIn(entry, ignored, f".gitignore is missing {entry}")

    def test_requirements_cover_the_imports(self):
        with open(p("requirements.txt")) as f:
            reqs = f.read().lower()
        for pkg in ("transformers", "bitsandbytes", "accelerate", "fastapi",
                    "uvicorn", "cryptography", "datasets", "huggingface_hub"):
            self.assertIn(pkg, reqs)

    def test_no_secrets_or_keys_committed(self):
        patterns = [r"hf_[A-Za-z0-9]{30,}", r"ghp_[A-Za-z0-9]{30,}",
                    r"sk-[A-Za-z0-9]{32,}", r"AIza[0-9A-Za-z_\-]{35}",
                    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"]
        bad_names = {"api_key.txt", "public_url.txt", ".env", "signing_key.pem"}
        problems = []
        for folder, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
            for name in files:
                if name in bad_names or name.endswith(".pem"):
                    problems.append(f"secret-like file: {name}")
                if name == os.path.basename(__file__):
                    continue   # this file contains the patterns themselves
                try:
                    with open(os.path.join(folder, name), encoding="utf-8") as f:
                        text = f.read()
                except (UnicodeDecodeError, OSError):
                    continue
                for pat in patterns:
                    if re.search(pat, text):
                        problems.append(f"{name}: matches {pat}")
        self.assertEqual(problems, [])

    def test_notebooks_have_no_widget_state(self):
        # Widget state makes GitHub show "Invalid Notebook".
        for name in os.listdir(ROOT):
            if name.endswith(".ipynb"):
                nb = load_json(name)
                self.assertNotIn("widgets", nb.get("metadata", {}), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)

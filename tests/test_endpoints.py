import time
import unittest
from unittest import mock

import tests  # noqa: F401
import quantize_service as qs
from verify_report import verify_report
from tests.sample_data import SAMPLE_MEMORY

SAFE = {"model_id": "org/model", "params_millions": 500.0, "flags": [], "safe_to_proceed": True}


def success(model_id="org/model"):
    return {"status": "SUCCESS", "model_id": model_id, "winner": "int8",
            "winner_stats": {"ppl_increase_pct": 1.0, "size_gb": 0.6}, "attempts": [],
            "artifact_sha256": "ab" * 32}


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.report = dict(SAFE)

        def fake_quantize(model_id, budget, bits, out_dir):
            self.calls.append((model_id, budget, bits, out_dir))
            return success(model_id)

        for target, value in (("fetch_model_report", lambda model_id: self.report),
                              ("quantize_smart", fake_quantize),
                              ("mem_load", lambda: SAMPLE_MEMORY)):
            patcher = mock.patch.object(qs, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def wait_for(self, job_id):
        for _ in range(200):
            if qs.JOBS[job_id]["status"] in ("DONE", "FAILED"):
                break
            time.sleep(0.02)
        return qs.JOBS[job_id]

    # ---- /quantize ----
    def test_unsafe_model_is_refused_without_touching_the_gpu(self):
        self.report = {"safe_to_proceed": False, "flags": ["TOO_LARGE: 7615.6M params > 3000M limit"]}
        out = qs.quantize_endpoint(qs.Req(model_id="org/huge"))
        self.assertEqual(out["status"], "REFUSED")
        self.assertEqual(out["flags"], ["TOO_LARGE: 7615.6M params > 3000M limit"])
        self.assertEqual(self.calls, [])
        self.assertNotIn("signature", out)

    def test_success_is_signed_and_uses_a_per_request_folder(self):
        out = qs.quantize_endpoint(qs.Req(model_id="org/model", accuracy_budget_pct=8.0))
        self.assertTrue(verify_report(out, qs._PUB_B64))
        model_id, budget, bits, out_dir = self.calls[0]
        self.assertEqual((model_id, budget, bits), ("org/model", 8.0, "auto"))
        self.assertEqual(out_dir, "/content/artifacts/org_model_8.0")

    def test_bits_parameter(self):
        for given, expected in (("4", 4), ("8", 8), ("auto", "auto"), ("16", "auto"), ("", "auto")):
            self.calls.clear()
            qs.quantize_endpoint(qs.Req(model_id="org/model", bits=given))
            self.assertEqual(self.calls[0][2], expected, given)

    # ---- /plan ----
    def test_plan_is_signed_and_correct(self):
        out = qs.plan_endpoint(qs.PlanReq(accuracy_budget_pct=8.0, max_size_gb=1.0))
        self.assertEqual(out["status"], "PLAN")
        self.assertTrue(verify_report(out, qs._PUB_B64))
        self.assertEqual(out["recommended"]["model_id"], "Qwen/Qwen2.5-0.5B")

    def test_plan_can_be_limited_to_listed_models(self):
        out = qs.plan_endpoint(qs.PlanReq(accuracy_budget_pct=20.0,
                                          model_ids=["microsoft/phi-2"]))
        self.assertEqual([f["model_id"] for f in out["fits"]], ["microsoft/phi-2"])

    # ---- /jobs ----
    def test_job_runs_to_completion_with_a_signed_result(self):
        job = qs.submit_job(qs.Req(model_id="org/model"))
        self.assertEqual(job["status"], "QUEUED")
        done = self.wait_for(job["job_id"])
        self.assertEqual(done["status"], "DONE")
        self.assertTrue(verify_report(done["result"], qs._PUB_B64))

    def test_job_for_a_refused_model_finishes_with_refusal(self):
        self.report = {"safe_to_proceed": False, "flags": ["GATED: needs license"]}
        done = self.wait_for(qs.submit_job(qs.Req(model_id="org/gated"))["job_id"])
        self.assertEqual(done["status"], "DONE")
        self.assertEqual(done["result"]["status"], "REFUSED")

    def test_job_failure_is_reported_not_raised(self):
        def explode(*args, **kwargs):
            raise RuntimeError("simulated crash")
        with mock.patch.object(qs, "quantize_smart", explode):
            done = self.wait_for(qs.submit_job(qs.Req(model_id="org/model"))["job_id"])
        self.assertEqual(done["status"], "FAILED")
        self.assertEqual(done["error"], "RuntimeError")

    def test_unknown_job_is_404(self):
        self.assertEqual(qs.job_status("does-not-exist").status_code, 404)


if __name__ == "__main__":
    unittest.main()

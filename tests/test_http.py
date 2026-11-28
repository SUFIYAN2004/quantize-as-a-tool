"""A few end-to-end checks through the real FastAPI routes and the key gate.

Skipped automatically when FastAPI's test client is not installed
(pip install -r requirements-dev.txt).
"""
import unittest
from unittest import mock

import tests  # noqa: F401
from tests.sample_data import SAMPLE_MEMORY

try:
    from starlette.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None

KEY = "test-key-0123456789abcdef"


@unittest.skipUnless(TestClient, "needs fastapi + httpx (pip install -r requirements-dev.txt)")
class HttpTests(unittest.TestCase):
    def setUp(self):
        import quantize_service as qs
        self.qs = qs
        self.client = TestClient(qs.KeyGate(qs.app, KEY))
        self.h = {"X-API-Key": KEY}

    def test_health_and_public_key_are_open(self):
        self.assertEqual(self.client.get("/health").json(), {"ok": True})
        r = self.client.get("/public-key")
        self.assertEqual(r.json()["public_key"], self.qs._PUB_B64)

    def test_protected_routes_reject_missing_and_wrong_keys(self):
        for method, path in (("post", "/plan"), ("post", "/quantize"), ("post", "/jobs"), ("get", "/jobs/x")):
            self.assertEqual(getattr(self.client, method)(path).status_code, 401, path)
            r = getattr(self.client, method)(path, headers={"X-API-Key": "nope"})
            self.assertEqual(r.status_code, 401, path)

    def test_plan_over_http(self):
        with mock.patch.object(self.qs, "mem_load", lambda: SAMPLE_MEMORY):
            r = self.client.post("/plan", headers=self.h,
                                 json={"accuracy_budget_pct": 8.0, "max_size_gb": 1.0})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["recommended"]["model_id"], "Qwen/Qwen2.5-0.5B")
        self.assertIn("signature", r.json())

    def test_refusal_over_http(self):
        unsafe = {"safe_to_proceed": False, "flags": ["GATED: needs license"]}
        with mock.patch.object(self.qs, "fetch_model_report", lambda model_id: unsafe):
            r = self.client.post("/quantize", headers=self.h, json={"model_id": "org/gated"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "REFUSED", "flags": ["GATED: needs license"]})

    def test_bad_request_body_is_rejected(self):
        r = self.client.post("/quantize", headers=self.h, json={"accuracy_budget_pct": 10})
        self.assertEqual(r.status_code, 422)

    def test_unknown_job_is_404(self):
        self.assertEqual(self.client.get("/jobs/nope", headers=self.h).status_code, 404)


if __name__ == "__main__":
    unittest.main()

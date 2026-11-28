import base64
import copy
import json
import unittest

import tests  # noqa: F401
import quantize_service as qs
from verify_report import verify_report
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

TRUSTED = qs._PUB_B64


def sample(status="SUCCESS"):
    return {"status": status, "model_id": "org/model", "winner": "int8",
            "winner_stats": {"ppl_increase_pct": 0.89, "size_gb": 0.63},
            "attempts": [{"config": "int8", "ppl_increase_pct": 0.89,
                          "size_gb": 0.63, "verdict": "PASS"}],
            "artifact_sha256": "ab" * 32}


def sign_with(body, key):
    """What an attacker with their own key could produce."""
    body = dict(body)
    body["signed_at"] = "2026-01-01T00:00:00+00:00"
    msg = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    pub = base64.b64encode(key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
    body["signature"] = {"alg": "ed25519", "public_key": pub,
                         "value": base64.b64encode(key.sign(msg)).decode()}
    return body, pub


class SigningTests(unittest.TestCase):
    def test_signed_report_verifies(self):
        rep = qs.sign_report(sample())
        self.assertTrue(verify_report(rep, TRUSTED))
        self.assertEqual(rep["signature"]["alg"], "ed25519")
        self.assertEqual(rep["signature"]["public_key"], TRUSTED)
        self.assertIn("signed_at", rep)

    def test_only_result_statuses_are_signed(self):
        for status in ("SUCCESS", "NO_CONFIG_MET_BUDGET", "PLAN"):
            self.assertIn("signature", qs.sign_report(sample(status)), status)
        for status in ("REFUSED", "ERROR", "UNAUTHORIZED"):
            self.assertNotIn("signature", qs.sign_report(sample(status)), status)

    def test_input_is_not_mutated(self):
        original = sample()
        snapshot = copy.deepcopy(original)
        qs.sign_report(original)
        self.assertEqual(original, snapshot)

    def test_resigning_replaces_the_old_signature(self):
        once = qs.sign_report(sample())
        twice = qs.sign_report(once)
        self.assertTrue(verify_report(twice, TRUSTED))

    def test_any_edit_breaks_the_signature(self):
        rep = qs.sign_report(sample())
        edits = [
            lambda r: r["winner_stats"].__setitem__("ppl_increase_pct", 0.1),
            lambda r: r["winner_stats"].__setitem__("size_gb", 0.3),
            lambda r: r.__setitem__("winner", "nf4"),
            lambda r: r.__setitem__("model_id", "evil/model"),
            lambda r: r["attempts"][0].__setitem__("verdict", "FAIL"),
            lambda r: r.__setitem__("artifact_sha256", "00" * 32),
            lambda r: r.__setitem__("signed_at", "2030-01-01T00:00:00+00:00"),
            lambda r: r.__setitem__("extra", "field"),
            lambda r: r.pop("winner"),
        ]
        for i, edit in enumerate(edits):
            forged = copy.deepcopy(rep)
            edit(forged)
            self.assertFalse(verify_report(forged, TRUSTED), f"edit {i} was accepted")

    def test_signature_cannot_be_moved_to_another_report(self):
        a = qs.sign_report(sample())
        b = sample()
        b["model_id"] = "other/model"
        b["signed_at"] = a["signed_at"]
        b["signature"] = a["signature"]
        self.assertFalse(verify_report(b, TRUSTED))

    def test_attacker_signature_fails_against_trusted_key(self):
        forged, attacker_pub = sign_with(sample(), Ed25519PrivateKey.generate())
        # Verifying against the key embedded in the report proves integrity only...
        self.assertTrue(verify_report(forged, attacker_pub))
        # ...which is why the verifier must be given a key it already trusts.
        self.assertFalse(verify_report(forged, TRUSTED))

    def test_attacker_claiming_the_trusted_key_fails(self):
        forged, _ = sign_with(sample(), Ed25519PrivateKey.generate())
        forged["signature"]["public_key"] = TRUSTED
        self.assertFalse(verify_report(forged, TRUSTED))

    def test_malformed_signatures_are_rejected_not_raised(self):
        rep = qs.sign_report(sample())
        cases = []
        for mutate in (
            lambda r: r.pop("signature"),
            lambda r: r.__setitem__("signature", None),
            lambda r: r.__setitem__("signature", {}),
            lambda r: r["signature"].__setitem__("value", "not base64!!"),
            lambda r: r["signature"].__setitem__("value", ""),
            lambda r: r["signature"].pop("value"),
        ):
            bad = copy.deepcopy(rep)
            mutate(bad)
            cases.append(bad)
        for bad in cases:
            self.assertFalse(verify_report(bad, TRUSTED))

    def test_garbage_trusted_key_is_rejected(self):
        rep = qs.sign_report(sample())
        self.assertFalse(verify_report(rep, "A" * 43 + "="))
        self.assertFalse(verify_report(rep, ""))


if __name__ == "__main__":
    unittest.main()

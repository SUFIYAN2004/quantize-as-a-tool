import json, base64
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def verify_report(report, trusted_public_key_b64):
    """True only if the report is unmodified AND signed by the trusted key.

    Fails closed: anything malformed (not a dict, odd signature block, bad base64,
    wrong key length) returns False instead of raising.
    """
    try:
        sig = report.get("signature")
        if not isinstance(sig, dict) or sig.get("public_key") != trusted_public_key_b64:
            return False
        body = {k: v for k, v in report.items() if k != "signature"}
        msg = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        Ed25519PublicKey.from_public_bytes(base64.b64decode(trusted_public_key_b64)).verify(
            base64.b64decode(sig["value"]), msg)
        return True
    except Exception:
        return False

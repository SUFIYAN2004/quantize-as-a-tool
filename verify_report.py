import json, base64
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

def verify_report(report, trusted_public_key_b64):
    """True only if the report is unmodified AND signed by the trusted key."""
    sig = report.get("signature") or {}
    if sig.get("public_key") != trusted_public_key_b64:
        return False
    body = {k: v for k, v in report.items() if k != "signature"}
    msg = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(trusted_public_key_b64)).verify(
            base64.b64decode(sig["value"]), msg)
        return True
    except (InvalidSignature, KeyError, ValueError):
        return False

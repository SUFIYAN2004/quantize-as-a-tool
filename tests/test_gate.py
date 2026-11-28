import asyncio
import unittest

import tests  # noqa: F401
import quantize_service as qs

KEY = "test-key-0123456789abcdef"


class Inner:
    """Stands in for the FastAPI app and records what reached it."""
    def __init__(self):
        self.paths = []

    async def __call__(self, scope, receive, send):
        self.paths.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def request(gate, path, key=None, scope_type="http"):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    headers = [(b"x-api-key", key.encode())] if key is not None else []
    scope = {"type": scope_type, "path": path, "method": "POST", "headers": headers}
    asyncio.run(gate(scope, receive, send))
    return sent[0].get("status")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.inner = Inner()
        self.gate = qs.KeyGate(self.inner, KEY)

    def test_protected_paths_need_the_key(self):
        for path in ("/quantize", "/plan", "/jobs", "/jobs/abc123"):
            self.assertEqual(request(self.gate, path), 401, path)
            self.assertEqual(request(self.gate, path, key="wrong-key"), 401, path)
            self.assertEqual(request(self.gate, path, key=""), 401, path)
        self.assertEqual(self.inner.paths, [])

    def test_correct_key_gets_through(self):
        for path in ("/quantize", "/plan", "/jobs", "/jobs/abc123"):
            self.assertEqual(request(self.gate, path, key=KEY), 200, path)
        self.assertEqual(len(self.inner.paths), 4)

    def test_key_prefix_and_extension_are_rejected(self):
        self.assertEqual(request(self.gate, "/plan", key=KEY[:-1]), 401)
        self.assertEqual(request(self.gate, "/plan", key=KEY + "x"), 401)

    def test_public_paths_stay_open(self):
        for path in ("/health", "/public-key", "/docs", "/openapi.json"):
            self.assertEqual(request(self.gate, path), 200, path)

    def test_non_http_scopes_pass_through(self):
        sent = []

        async def receive():
            return {"type": "lifespan.startup"}

        async def send(message):
            sent.append(message)

        asyncio.run(self.gate({"type": "lifespan", "path": "/quantize", "headers": []}, receive, send))
        self.assertEqual(self.inner.paths, ["/quantize"])


if __name__ == "__main__":
    unittest.main()

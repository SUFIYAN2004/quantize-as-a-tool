"""Test setup: no GPU, no network, no model downloads.

The heavy libraries (torch, transformers, huggingface_hub, datasets) are replaced with
mocks BEFORE quantize_service is imported, and its state files point at a temp directory,
so running the tests never touches your real signing key or measurement cache.
"""
import os
import sys
import tempfile
from unittest import mock

TMP_DIR = tempfile.mkdtemp(prefix="quantize-tests-")
os.environ["MEM_PATH"] = os.path.join(TMP_DIR, "memory.json")
os.environ["SIGNING_KEY_PATH"] = os.path.join(TMP_DIR, "signing_key.pem")

for _name in ("torch", "transformers", "huggingface_hub", "datasets"):
    sys.modules[_name] = mock.MagicMock(name=_name)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

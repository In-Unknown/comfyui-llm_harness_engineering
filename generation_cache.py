"""Session-only result reuse shared by all generative nodes in this pack."""

import hashlib
from io import BytesIO
import json
from threading import Lock
from uuid import uuid4

import numpy as np
from PIL import Image
from comfy_api.latest import io
from comfy.model_management import throw_exception_if_processing_interrupted


class CachedGenerationNode(io.ComfyNode):
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # ComfyUI must consult this session's cache and current connection settings every run.
        return float("nan")


def _input_identity(value):
    if value is None or isinstance(value, (str, bool, int)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, dict):
        return ["dict", [[name, _input_identity(value[name])] for name in sorted(value)]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_input_identity(item) for item in value]]
    if isinstance(value, np.ndarray) or hasattr(value, "detach"):
        throw_exception_if_processing_interrupted()
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        digest = hashlib.sha256(json.dumps([array.shape, array.dtype.str]).encode("utf-8"))
        digest.update(np.ascontiguousarray(array))
        return ["array", digest.hexdigest()]
    raise TypeError(f"Unsupported generation cache input: {type(value).__name__}")


def input_key(node_type, inputs):
    """Hash named inputs, including exact tensor contents, independently of node IDs."""
    value = [node_type, _input_identity(inputs)]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


class GenerationCache:
    def __init__(self):
        self.session_id = uuid4().hex
        self._results = {}
        self._key_locks = {}
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            return self._results.get(key)

    def run(self, key, generate, *, force_refresh=False):
        """Return (immutable result, reused); commit only completed, uninterrupted work.

        Callbacks return text, bytes or tuples of immutable values, never live tensors.
        Per-input locks let unrelated requests proceed concurrently.
        """
        with self._lock:
            key_lock = self._key_locks.setdefault(key, Lock())
        while not key_lock.acquire(timeout=0.1):
            throw_exception_if_processing_interrupted()
        try:
            throw_exception_if_processing_interrupted()
            with self._lock:
                if not force_refresh and key in self._results:
                    return self._results[key], True
            result = generate()
            throw_exception_if_processing_interrupted()
            with self._lock:
                self._results[key] = result
            return result, False
        finally:
            key_lock.release()


CACHE = GenerationCache()


def decode_image(content):
    # Each consumer owns a fresh image; downstream edits cannot change the cached PNG.
    with Image.open(BytesIO(content)) as image:
        return image.convert("RGB")


def cached_image(key, generate, *, force_refresh=False):
    def encode():
        image = generate()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    content, reused = CACHE.run(key, encode, force_refresh=force_refresh)
    return decode_image(content), reused

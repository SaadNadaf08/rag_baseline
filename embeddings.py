"""Pluggable embedders. `st` is the default; `hash` is a deterministic
toy embedder so you can test the whole pipeline offline with zero installs.
Swap embedders freely - everything downstream only calls embed()."""
import hashlib
import re

import numpy as np
import requests

from config import settings

_TOKEN = re.compile(r"[a-z0-9']+")


class HashingEmbedder:
    """Test-only bag-of-words hashing embedder. No model download needed."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _TOKEN.findall(t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vecs[i, h % self.dimension] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class STEmbedder:
    """sentence-transformers, runs locally after a one-time model download."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=settings.embed_batch_size,
                                            show_progress_bar=False), dtype=np.float32)


class OpenAIEmbedder:
    """Any OpenAI-compatible /embeddings endpoint (works with LiteLLM later)."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model, self.api_key, self.base_url = model, api_key, base_url
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        import time
        out, batch = [], 100  # API batch limit
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            r = requests.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": part}, timeout=120)
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out.extend(np.asarray([d["embedding"] for d in data], dtype=np.float32))
            time.sleep(0.1)
        return np.vstack(out)


def get_embedder():
    kind = settings.embedder.lower()
    if kind == "st":
        return STEmbedder(settings.embed_model)
    if kind == "openai":
        import os
        key = os.environ["OPENAI_API_KEY"]
        return OpenAIEmbedder("text-embedding-3-small", key, settings.llm_base_url)
    if kind == "hash":
        print("[warn] hash embedder is for offline testing only - not a real baseline")
        return HashingEmbedder()
    raise ValueError(f"unknown embedder: {kind}")

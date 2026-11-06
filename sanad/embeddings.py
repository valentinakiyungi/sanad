"""Dense embeddings: Gemini in production, a hashing embedder offline.

``HashEmbedder`` needs no network or model download, so tests, CI and the
offline eval run anywhere. It captures character n-gram overlap within one
language but cannot match Arabic to English — which is exactly the gap the
knowledge graph's language-twin edges close (see README results).
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .arabic import normalize, tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], task: str = "document") -> np.ndarray: ...


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class HashEmbedder:
    """Signed feature hashing of stemmed words + character 3/4-grams."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> list[str]:
        words = tokenize(text)
        feats = [f"w:{w}" for w in words]
        for w in normalize(text).split():
            padded = f"#{w}#"
            for n in (3, 4):
                feats += [f"c{n}:{padded[i : i + n]}" for i in range(len(padded) - n + 1)]
        return feats

    def embed(self, texts: list[str], task: str = "document") -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f in self._features(text):
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                m[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _unit(m)


class GeminiEmbedder:
    """Multilingual embeddings via the Gemini API (Arabic <-> English aware)."""

    def __init__(self, client, model: str, dim: int = 768, batch: int = 50):
        self.client, self.model, self.dim, self.batch = client, model, dim, batch
        self.name = f"gemini:{model}:{dim}"

    def embed(self, texts: list[str], task: str = "document") -> np.ndarray:
        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            rows += self.client.embed(texts[i : i + self.batch], self.model, task_type, self.dim)
        return _unit(np.asarray(rows, dtype=np.float32))

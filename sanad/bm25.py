"""Okapi BM25 over Arabic-normalised, lightly stemmed tokens.

Implemented directly (a few dozen lines) so the Arabic tokeniser is used for
both documents and queries — off-the-shelf BM25 libraries split on whitespace
and miss most Arabic matches because of attached prefixes.
"""

from __future__ import annotations

import math
from collections import Counter

from .arabic import tokenize


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in terms:
                tf = doc.get(t)
                if not tf:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                s += self.idf[t] * tf * (self.k1 + 1) / norm
            out.append(s)
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked[:k] if s > 0]

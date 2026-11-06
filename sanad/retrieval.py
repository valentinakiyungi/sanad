"""Hybrid retrieval (BM25 + dense, fused with RRF) plus knowledge-graph expansion.

Retrieval modes (used by the eval ablation):
  bm25    lexical only, Arabic-aware tokeniser
  dense   embeddings only
  hybrid  reciprocal rank fusion of bm25 + dense
  graph   hybrid, then expand the top seeds through the knowledge graph:
          - REFERENCES edges  ("subject to Article (8)")  -> multi-hop answers
          - language twins    (same article, other language) -> cross-lingual
          - DEFINES edges     (definition questions)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .arabic import detect_language
from .bm25 import BM25
from .corpus import Chunk
from .embeddings import Embedder
from .kg import LegalGraph

MODES = ("bm25", "dense", "hybrid", "graph")
_DEFINITION_INTENT = re.compile(
    r"\b(mean|means|meaning|definition|define|defined)\b|ما معنى|ما المقصود|تعريف|يقصد",
    re.IGNORECASE,
)


@dataclass
class Hit:
    index: int
    score: float
    via: list[str] = field(default_factory=list)


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: robust to the very different score scales of BM25 and cosine."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder, graph: LegalGraph):
        self.chunks, self.vectors, self.embedder, self.graph = chunks, vectors, embedder, graph
        self.bm25 = BM25([f"{c.title}\n{c.text}" for c in chunks])

    # --- build / persist ------------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> Index:
        vectors = embedder.embed([f"{c.title}\n{c.text}" for c in chunks], task="document")
        return cls(chunks, vectors, embedder, LegalGraph.build(chunks))

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(d / "vectors.npy", self.vectors)
        self.graph.save(d / "graph.json")
        (d / "meta.json").write_text(json.dumps({"embedder": self.embedder.name, "chunks": len(self.chunks)}))

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> Index:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        if meta["embedder"] != embedder.name:
            raise ValueError(
                f"Index was built with '{meta['embedder']}' but '{embedder.name}' is configured. "
                "Rebuild it with `python -m sanad index`."
            )
        chunks = [Chunk(**json.loads(line)) for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        return cls(chunks, np.load(d / "vectors.npy"), embedder, LegalGraph.load(d / "graph.json"))

    # --- retrieval ------------------------------------------------------------------
    def search_bm25(self, query: str, k: int = 20) -> list[int]:
        return [i for i, _ in self.bm25.search(query, k)]

    def search_dense(self, query: str, k: int = 20) -> list[int]:
        q = self.embedder.embed([query], task="query")[0]
        sims = self.vectors @ q
        return [int(i) for i in np.argsort(-sims)[:k] if sims[i] > 0]

    def retrieve(self, query: str, mode: str = "graph", k: int = 8, extra_queries: list[str] | None = None) -> list[Hit]:
        """Return up to ``k`` chunk hits for a question (plus optional rewritten queries)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        queries = [query, *(extra_queries or [])]
        rankings: list[list[int]] = []
        sources: dict[int, set[str]] = {}
        for q in queries:
            if mode in ("bm25", "hybrid", "graph"):
                r = self.search_bm25(q)
                rankings.append(r)
                for i in r:
                    sources.setdefault(i, set()).add("bm25")
            if mode in ("dense", "hybrid", "graph"):
                r = self.search_dense(q)
                rankings.append(r)
                for i in r:
                    sources.setdefault(i, set()).add("dense")
        fused = rrf(rankings)
        hits = {i: Hit(i, s, sorted(sources[i])) for i, s in fused.items()}
        if mode == "graph":
            self._expand(hits, query)
        return sorted(hits.values(), key=lambda h: -h.score)[:k]

    def _expand(self, hits: dict[int, Hit], query: str, seeds: int = 3) -> None:
        if not hits:
            return
        lang = detect_language(query)
        ranked = sorted(hits.values(), key=lambda h: -h.score)
        top = ranked[0].score

        def add(idx: int, score: float, via: str) -> None:
            if idx in hits:
                hits[idx].score = max(hits[idx].score, score)
                if via not in hits[idx].via:
                    hits[idx].via.append(via)
            else:
                hits[idx] = Hit(idx, score, [via])

        def add_article(article_id: str, score: float, via: str) -> None:
            for chunk_lang, idx in self.graph.chunks_of(article_id).items():
                add(idx, score if chunk_lang == lang else score * 0.97, via)

        # RRF scores are compressed (rank 2 is ~0.98x rank 1), so fixed multipliers
        # would never surface a referenced article. Instead: an article cited by the
        # best match is ranked directly after it — in legal text, "subject to
        # Article (8)" means Article (8) is part of the answer. References of the
        # other seeds get a smaller boost.
        seen_articles: list[str] = []
        for h in ranked:
            aid = self.chunks[h.index].article_id
            if aid in seen_articles:
                continue
            is_best = not seen_articles
            seen_articles.append(aid)
            for idx in self.graph.chunks_of(aid).values():
                if idx != h.index:
                    add(idx, h.score * 0.999, "graph:language-twin")
            for ref in self.graph.references(aid):
                add_article(ref, h.score * (0.995 if is_best else 0.9), "graph:references")
            for ref in self.graph.referenced_by(aid):
                add_article(ref, h.score * 0.5, "graph:referenced-by")
            if len(seen_articles) >= seeds:
                break

        if _DEFINITION_INTENT.search(query):
            for term in self.graph.terms_in(query):
                for aid in self.graph.defining_articles(term):
                    add_article(aid, top * 1.01, "graph:defines")


def article_ranking(index: Index, hits: list[Hit]) -> list[str]:
    """Collapse chunk hits to language-independent article ids, preserving order."""
    out: list[str] = []
    for h in hits:
        aid = index.chunks[h.index].article_id
        if aid not in out:
            out.append(aid)
    return out

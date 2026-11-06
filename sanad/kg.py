"""Legal knowledge graph built from the corpus.

Nodes
  law:<LAW-ID>               a regulation
  article:<LAW-ID>:<n>       an article, language-independent
  term:<normalised term>     a defined term ("Wage", «الأجر»)

Edges
  CONTAINS    law -> article
  REFERENCES  article -> article     ("subject to Article (8)")
  DEFINES     article -> term
  MENTIONS    article -> term

Each article node keeps the indices of its Arabic and English chunks, which is
what lets retrieval jump languages ("language twins") without translation.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from .arabic import normalize
from .corpus import Chunk


class LegalGraph:
    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.g = graph if graph is not None else nx.MultiDiGraph()

    @classmethod
    def build(cls, chunks: list[Chunk]) -> LegalGraph:
        kg = cls()
        g = kg.g
        for i, c in enumerate(chunks):
            law, art = f"law:{c.law_id}", f"article:{c.article_id}"
            if law not in g:
                g.add_node(law, kind="law", titles={})
            g.nodes[law]["titles"][c.lang] = c.law_title
            if "chunks" not in g.nodes.get(art, {}):  # may already exist as a reference target
                g.add_node(art, kind="article", law=c.law_id, number=c.article, chunks={})
                g.add_edge(law, art, key="CONTAINS")
            g.nodes[art]["chunks"][c.lang] = i
            if c.lang == "en" or "title" not in g.nodes[art]:
                g.nodes[art]["title"] = c.title
            for ref in c.references:
                target = f"article:{c.law_id}:{ref}"
                if not g.has_edge(art, target, key="REFERENCES"):
                    g.add_edge(art, target, key="REFERENCES")
            for term in c.defines:
                t = f"term:{normalize(term)}"
                g.add_node(t, kind="term", label=term)
                g.add_edge(art, t, key="DEFINES")
        # MENTIONS: which articles use each defined term.
        terms = [(n, n[len("term:") :]) for n, d in g.nodes(data=True) if d.get("kind") == "term"]
        for c in chunks:
            text, art = normalize(c.text), f"article:{c.article_id}"
            for node, norm_term in terms:
                linked = g.has_edge(art, node, key="DEFINES") or g.has_edge(art, node, key="MENTIONS")
                if norm_term in text and not linked:
                    g.add_edge(art, node, key="MENTIONS")
        return kg

    # --- queries used by retrieval -------------------------------------------------
    def chunks_of(self, article_id: str) -> dict[str, int]:
        node = f"article:{article_id}"
        return dict(self.g.nodes[node]["chunks"]) if node in self.g else {}

    def references(self, article_id: str) -> list[str]:
        node = f"article:{article_id}"
        if node not in self.g:
            return []
        # Skip references to articles that are not in the corpus (e.g. a truncated PDF).
        return [
            v[len("article:") :]
            for _, v, k in self.g.out_edges(node, keys=True)
            if k == "REFERENCES" and "chunks" in self.g.nodes[v]
        ]

    def referenced_by(self, article_id: str) -> list[str]:
        node = f"article:{article_id}"
        if node not in self.g:
            return []
        return [u[len("article:") :] for u, _, k in self.g.in_edges(node, keys=True) if k == "REFERENCES"]

    def terms_in(self, query: str) -> list[str]:
        q = normalize(query)
        return [n for n, d in self.g.nodes(data=True) if d.get("kind") == "term" and n[len("term:") :] in q]

    def defining_articles(self, term_node: str) -> list[str]:
        return [u[len("article:") :] for u, _, k in self.g.in_edges(term_node, keys=True) if k == "DEFINES"]

    def stats(self) -> dict[str, int]:
        kinds: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
        edges: dict[str, int] = {}
        for _, _, k in self.g.edges(keys=True):
            edges[k] = edges.get(k, 0) + 1
        return {**{f"nodes_{k}": v for k, v in kinds.items()}, **{f"edges_{k}": v for k, v in edges.items()}}

    # --- persistence ----------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.write_text(json.dumps(nx.node_link_data(self.g, edges="edges"), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LegalGraph:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(nx.node_link_graph(data, directed=True, multigraph=True, edges="edges"))

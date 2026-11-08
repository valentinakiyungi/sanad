"""The question-answering pipeline as a graph of small, testable nodes.

    detect_language -> rewrite_query -> retrieve --(no hits)--> validate
                                            |
                                            v
                                     select_context -> generate -> validate

The same node functions run in two ways:
  * as a LangGraph ``StateGraph`` (production / API) when ``langgraph`` is installed
  * with a tiny sequential runner, so tests and CI need no extra dependencies
"""

from __future__ import annotations

import operator
import re
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from .arabic import detect_language
from .llm import LLM
from .prompts import ABSTAIN, ANSWER_SYSTEM, LANG_NAME, REWRITE_SYSTEM, answer_prompt, rewrite_prompt
from .retrieval import Hit, Index

_MARKER = re.compile(r"\[S(\d+)\]")


class State(TypedDict, total=False):
    question: str
    lang: str
    queries: list[str]
    hits: list[Hit]
    sources: list[dict]
    draft: str
    answer: str
    citations: list[dict]
    abstained: bool
    trace: Annotated[list[dict], operator.add]


Node = Callable[[State], dict]


def _timed(name: str, fn: Node) -> Node:
    def wrapper(state: State) -> dict:
        t0 = time.perf_counter()
        update = fn(state)
        note = update.pop("_note", "")
        update["trace"] = [{"node": name, "ms": round((time.perf_counter() - t0) * 1000, 1), "note": note}]
        return update

    return wrapper


class Sanad:
    """Bilingual GraphRAG question answering over an :class:`Index`."""

    def __init__(
        self,
        index: Index,
        llm: LLM,
        mode: str = "graph",
        k: int = 8,
        max_articles: int = 4,
        use_langgraph: bool = True,
    ):
        self.index, self.llm, self.mode, self.k, self.max_articles = index, llm, mode, k, max_articles
        self.nodes: dict[str, Node] = {
            "detect_language": _timed("detect_language", self.detect_language),
            "rewrite_query": _timed("rewrite_query", self.rewrite_query),
            "retrieve": _timed("retrieve", self.retrieve),
            "select_context": _timed("select_context", self.select_context),
            "generate": _timed("generate", self.generate),
            "validate": _timed("validate", self.validate),
        }
        self.app = self._build_langgraph() if use_langgraph else None

    # --- nodes ------------------------------------------------------------------------
    def detect_language(self, state: State) -> dict:
        lang = detect_language(state["question"])
        return {"lang": lang, "_note": lang}

    def rewrite_query(self, state: State) -> dict:
        """Translate the question so BM25 can match articles in the other language too."""
        target = "en" if state["lang"] == "ar" else "ar"
        try:
            text = self.llm.generate(REWRITE_SYSTEM, rewrite_prompt(state["question"], target), temperature=0.0)
        except Exception as e:  # rewriting is an optimisation: never fail the request on it
            return {"queries": [], "_note": f"skipped ({type(e).__name__})"}
        line = text.strip().splitlines()[0].strip() if text.strip() else ""
        return {"queries": [line] if line else [], "_note": line or "none"}

    def retrieve(self, state: State) -> dict:
        if not self.index.is_relevant(state["question"], state.get("queries")):
            return {"hits": [], "_note": "no relevant articles (gate)"}
        hits = self.index.retrieve(state["question"], mode=self.mode, k=self.k, extra_queries=state.get("queries"))
        via_graph = sum(1 for h in hits if any(v.startswith("graph:") for v in h.via))
        return {"hits": hits, "_note": f"{len(hits)} chunks, {via_graph} via graph"}

    def select_context(self, state: State) -> dict:
        """One chunk per article (prefer the question's language), up to max_articles."""
        lang = state["lang"]
        chosen: dict[str, Hit] = {}
        for h in state["hits"]:
            c = self.index.chunks[h.index]
            current = chosen.get(c.article_id)
            if current is None:
                if len(chosen) < self.max_articles:
                    chosen[c.article_id] = h
            elif self.index.chunks[current.index].lang != lang and c.lang == lang:
                chosen[c.article_id] = Hit(h.index, current.score, sorted(set(current.via + h.via)))
        sources = []
        for n, h in enumerate(chosen.values(), start=1):
            c = self.index.chunks[h.index]
            sources.append(
                {
                    "marker": f"S{n}",
                    "chunk_id": c.chunk_id,
                    "article_id": c.article_id,
                    "label": c.label(),
                    "title": c.title,
                    "lang": c.lang,
                    "text": c.text,
                    "score": round(h.score, 5),
                    "via": h.via,
                }
            )
        return {"sources": sources, "_note": ", ".join(s["article_id"] for s in sources)}

    def generate(self, state: State) -> dict:
        system = ANSWER_SYSTEM.format(language=LANG_NAME[state["lang"]])
        draft = self.llm.generate(system, answer_prompt(state["question"], state["sources"]))
        return {"draft": draft}

    def validate(self, state: State) -> dict:
        """Drop citations to sources that were never provided; abstain if nothing is grounded."""
        lang = state.get("lang", "en")
        sources = {s["marker"]: s for s in state.get("sources", [])}
        draft = state.get("draft", "")
        valid, invalid = [], []
        for m in _MARKER.findall(draft):
            (valid if f"S{m}" in sources else invalid).append(f"S{m}")
        answer = _MARKER.sub(lambda m: m.group(0) if f"S{m.group(1)}" in sources else "", draft).strip()
        if not valid or "NOT_FOUND" in draft:
            return {"answer": ABSTAIN[lang], "citations": [], "abstained": True, "_note": "abstained"}
        ordered = list(dict.fromkeys(valid))
        citations = [{k: sources[m][k] for k in ("marker", "label", "title", "article_id", "lang")} for m in ordered]
        return {
            "answer": re.sub(r"\s{2,}", " ", answer),
            "citations": citations,
            "abstained": False,
            "_note": f"{len(ordered)} valid, {len(invalid)} removed",
        }

    # --- orchestration ------------------------------------------------------------------
    @staticmethod
    def _after_retrieve(state: State) -> str:
        return "select_context" if state.get("hits") else "validate"

    def _build_langgraph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None
        g = StateGraph(State)
        for name, fn in self.nodes.items():
            g.add_node(name, fn)
        g.add_edge(START, "detect_language")
        g.add_edge("detect_language", "rewrite_query")
        g.add_edge("rewrite_query", "retrieve")
        g.add_conditional_edges("retrieve", self._after_retrieve, ["select_context", "validate"])
        g.add_edge("select_context", "generate")
        g.add_edge("generate", "validate")
        g.add_edge("validate", END)
        return g.compile()

    def _run_sequential(self, state: State) -> State:
        order = ["detect_language", "rewrite_query", "retrieve"]
        for name in order:
            state = self._merge(state, self.nodes[name](state))
        rest = ["select_context", "generate", "validate"] if self._after_retrieve(state) == "select_context" else ["validate"]
        for name in rest:
            state = self._merge(state, self.nodes[name](state))
        return state

    @staticmethod
    def _merge(state: State, update: dict) -> State:
        merged = dict(state)
        for key, value in update.items():
            merged[key] = merged.get(key, []) + value if key == "trace" else value
        return merged  # type: ignore[return-value]

    def ask(self, question: str) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("question is empty")
        init: State = {"question": question, "trace": []}
        state = self.app.invoke(init) if self.app is not None else self._run_sequential(init)
        return {
            "question": question,
            "language": state["lang"],
            "answer": state["answer"],
            "abstained": state["abstained"],
            "citations": state["citations"],
            "sources": state.get("sources", []),
            "rewritten_queries": state.get("queries", []),
            "trace": state["trace"],
            "engine": "langgraph" if self.app is not None else "sequential",
            "model": self.llm.name,
            "retrieval_mode": self.mode,
        }

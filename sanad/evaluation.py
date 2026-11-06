"""Retrieval ablation: how much do hybrid fusion and the graph each add?

Metrics are computed on language-independent article ids, so a question in
Arabic is scored the same whether it finds the Arabic or English text.

  recall@k     share of gold articles found in the top k
  complete@k   share of questions with ALL gold articles in the top k
               (the multi-hop metric: a half-retrieved answer is a wrong answer)
  mrr          mean reciprocal rank of the first gold article
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .retrieval import MODES, Index, article_ranking


def load_gold(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def score_question(ranking: list[str], gold: list[str], k: int) -> dict[str, float]:
    top = ranking[:k]
    found = sum(1 for g in gold if g in top)
    rr = next((1.0 / (r + 1) for r, a in enumerate(ranking) if a in gold), 0.0)
    return {"recall": found / len(gold), "complete": float(found == len(gold)), "mrr": rr}


def evaluate(index: Index, gold: list[dict], k: int = 3, modes: tuple[str, ...] = MODES) -> dict:
    results: dict[str, dict] = {}
    for mode in modes:
        groups: dict[str, list[dict]] = defaultdict(list)
        misses = []
        for q in gold:
            ranking = article_ranking(index, index.retrieve(q["question"], mode=mode, k=20))
            s = score_question(ranking, q["gold"], k)
            for g in ("all", q["type"], q["lang"]):
                groups[g].append(s)
            if s["complete"] < 1:
                misses.append({"id": q["id"], "gold": q["gold"], "got": ranking[:k]})
        results[mode] = {
            g: {m: round(sum(x[m] for x in rows) / len(rows), 3) for m in ("recall", "complete", "mrr")} | {"n": len(rows)}
            for g, rows in groups.items()
        }
        results[mode]["misses"] = misses
    return {"k": k, "embedder": index.embedder.name, "questions": len(gold), "modes": results}


def to_markdown(report: dict) -> str:
    k = report["k"]
    lines = [
        f"Embedder: `{report['embedder']}` · {report['questions']} questions · k = {k}",
        "",
        f"| Mode | Recall@{k} | Complete@{k} (all) | Complete@{k} (multi-hop) | MRR | Recall@{k} (Arabic) |",
        "|---|---|---|---|---|---|",
    ]
    for mode, r in report["modes"].items():
        mh = r.get("multi_hop", {}).get("complete", float("nan"))
        ar = r.get("ar", {}).get("recall", float("nan"))
        lines.append(
            f"| {mode} | {r['all']['recall']:.2f} | {r['all']['complete']:.2f} | {mh:.2f} | {r['all']['mrr']:.2f} | {ar:.2f} |"
        )
    return "\n".join(lines)

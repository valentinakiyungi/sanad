"""Command line: build an index, ask questions, run the evaluation.

python -m sanad index                    # build .index from data/demo
python -m sanad ask "ما هي مدة فترة التجربة؟"
python -m sanad eval --out eval/results.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .corpus import load_corpus
from .evaluation import evaluate, load_gold, to_markdown
from .pipeline import Sanad
from .retrieval import MODES, Index


def _load_or_build(settings: Settings) -> Index:
    embedder = settings.make_embedder()
    if (Path(settings.index_dir) / "meta.json").exists():
        try:
            return Index.load(settings.index_dir, embedder)
        except ValueError as e:
            print(f"note: {e} Rebuilding.", file=sys.stderr)
    index = Index.build(load_corpus(settings.corpus_dir), embedder)
    index.save(settings.index_dir)
    return index


def cmd_index(args: argparse.Namespace, s: Settings) -> None:
    corpus = args.corpus or s.corpus_dir
    index = Index.build(load_corpus(corpus), s.make_embedder())
    index.save(args.out or s.index_dir)
    stats = index.graph.stats()
    print(f"Indexed {len(index.chunks)} article chunks from {corpus} with {index.embedder.name}")
    print("Graph:", ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))


def cmd_ask(args: argparse.Namespace, s: Settings) -> None:
    sanad = Sanad(_load_or_build(s), s.make_llm(), mode=args.mode or s.mode)
    result = sanad.ask(args.question)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    print(result["answer"], "\n")
    for c in result["citations"]:
        print(f"  [{c['marker']}] {c['label']} — {c['title']}")
    if args.trace:
        print()
        for t in result["trace"]:
            print(f"  {t['node']:<16}{t['ms']:>8} ms  {t['note']}")


def cmd_eval(args: argparse.Namespace, s: Settings) -> None:
    index = _load_or_build(s)
    report = evaluate(index, load_gold(args.gold), k=args.k)
    md = to_markdown(report)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n", encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sanad", description="Bilingual Arabic-English GraphRAG over regulations")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="build the search index and knowledge graph")
    pi.add_argument("--corpus", help="folder of <law>.<lang>.md|txt|pdf files")
    pi.add_argument("--out", help="index directory")

    pa = sub.add_parser("ask", help="ask a question in Arabic or English")
    pa.add_argument("question")
    pa.add_argument("--mode", choices=MODES)
    pa.add_argument("--trace", action="store_true", help="show pipeline timings")
    pa.add_argument("--json", action="store_true")

    pe = sub.add_parser("eval", help="retrieval ablation on the gold question set")
    pe.add_argument("--gold", default="eval/gold.jsonl")
    pe.add_argument("--k", type=int, default=3)
    pe.add_argument("--out")

    args = p.parse_args(argv)
    settings = Settings.from_env()
    {"index": cmd_index, "ask": cmd_ask, "eval": cmd_eval}[args.cmd](args, settings)


if __name__ == "__main__":
    main()

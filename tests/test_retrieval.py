import pytest

from sanad.embeddings import HashEmbedder
from sanad.evaluation import evaluate, load_gold, score_question
from sanad.retrieval import Index, article_ranking, rrf

from .conftest import ROOT


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert fused[1] == fused[2] > fused[3] > 0
    assert 4 in fused


def test_arabic_bm25_finds_article(index):
    top = article_ranking(index, index.retrieve("كم رسوم السداد المبكر للقرض؟", mode="bm25"))
    assert top[0] == "DCFR-2026:5"


def test_graph_adds_referenced_article(index):
    q = "If my contract is ended during probation, do I get the end-of-service gratuity?"
    hybrid = article_ranking(index, index.retrieve(q, mode="hybrid"))[:3]
    graph_hits = index.retrieve(q, mode="graph")
    graph = article_ranking(index, graph_hits)[:3]
    assert {"DER-2026:3", "DER-2026:8"} <= set(graph)
    assert any("graph:references" in h.via for h in graph_hits)
    assert len(set(graph) & {"DER-2026:3", "DER-2026:8"}) >= len(set(hybrid) & {"DER-2026:3", "DER-2026:8"})


def test_language_twin_brings_other_language(index):
    hits = index.retrieve("ما هي النسبة القصوى لعبء الدين؟", mode="graph")
    langs = {index.chunks[h.index].lang for h in hits if index.chunks[h.index].article_id == "DCFR-2026:3"}
    assert langs == {"ar", "en"}


def test_definition_intent_routes_to_definitions(index):
    top = article_ranking(index, index.retrieve("ما معنى القرض الشخصي؟", mode="graph"))
    assert top[0] == "DCFR-2026:1"


def test_invalid_mode(index):
    with pytest.raises(ValueError):
        index.retrieve("x", mode="nope")


def test_score_question():
    s = score_question(["a", "b", "c"], ["b", "z"], k=3)
    assert s == {"recall": 0.5, "complete": 0.0, "mrr": 0.5}


def test_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = Index.load(tmp_path, HashEmbedder())
    q = "How much is overtime paid at night?"
    assert article_ranking(loaded, loaded.retrieve(q)) == article_ranking(index, index.retrieve(q))


def test_load_rejects_mismatched_embedder(index, tmp_path):
    index.save(tmp_path)
    with pytest.raises(ValueError, match="Rebuild"):
        Index.load(tmp_path, HashEmbedder(dim=256))


def test_graph_mode_is_not_worse_than_hybrid(index):
    """Regression guard for the README results table."""
    report = evaluate(index, load_gold(ROOT / "eval" / "gold.jsonl"), k=3)
    modes = report["modes"]
    assert modes["graph"]["multi_hop"]["complete"] >= modes["hybrid"]["multi_hop"]["complete"]
    assert modes["graph"]["all"]["recall"] >= modes["hybrid"]["all"]["recall"]

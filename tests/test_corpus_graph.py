from sanad.corpus import find_references, parse_document


def test_loads_both_languages_for_every_article(chunks):
    ids = {c.chunk_id for c in chunks}
    assert len(chunks) == 32
    for law in ("DER-2026", "DCFR-2026"):
        for n in range(1, 9):
            assert f"{law}:{n}:ar" in ids and f"{law}:{n}:en" in ids


def test_references_match_across_languages(chunks):
    by_id = {c.chunk_id: c for c in chunks}
    for c in chunks:
        if c.lang == "en":
            assert c.references == by_id[f"{c.law_id}:{c.article}:ar"].references, c.chunk_id


def test_reference_patterns():
    assert find_references("subject to Article (8)") == [8]
    assert find_references("under Articles (3) and (4)") == [3, 4]
    assert find_references("وفقاً للمادة (٨)") == [8]
    assert find_references("أحكام المادتين (3) و(4)") == [3, 4]
    assert find_references("no references here") == []


def test_parse_handles_unbracketed_headers():
    chunks = parse_document("# Law\nArticle 1 - Scope\nApplies to all.\nArticle 2: Fees\nSee Article 1.", "X")
    assert [(c.article, c.title) for c in chunks] == [(1, "Scope"), (2, "Fees")]
    assert chunks[1].references == [1]


def test_definitions_extracted(chunks):
    der1 = next(c for c in chunks if c.chunk_id == "DER-2026:1:ar")
    assert "الأجر" in der1.defines


def test_graph_structure(index):
    g = index.graph
    assert set(g.chunks_of("DER-2026:8")) == {"ar", "en"}
    assert g.references("DER-2026:8") == ["DER-2026:3"]
    assert "DER-2026:3" in g.referenced_by("DER-2026:8")
    stats = g.stats()
    assert stats["nodes_article"] == 16 and stats["edges_REFERENCES"] == 8


def test_graph_roundtrip(index, tmp_path):
    from sanad.kg import LegalGraph

    index.graph.save(tmp_path / "g.json")
    loaded = LegalGraph.load(tmp_path / "g.json")
    assert loaded.references("DCFR-2026:4") == ["DCFR-2026:3"]
    assert loaded.chunks_of("DCFR-2026:4") == index.graph.chunks_of("DCFR-2026:4")

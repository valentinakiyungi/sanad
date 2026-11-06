import pytest

from sanad.pipeline import Sanad
from sanad.prompts import ABSTAIN


class ScriptedLLM:
    """Returns a fixed answer, to test citation validation in isolation."""

    name = "scripted"

    def __init__(self, answer: str, rewrite: str = ""):
        self.answer, self.rewrite = answer, rewrite

    def generate(self, system, prompt, temperature=0.1):
        return self.rewrite if "REWRITE" in system else self.answer


def test_end_to_end_arabic(sanad):
    r = sanad.ask("كم تخفض ساعات العمل في رمضان؟")
    assert r["language"] == "ar"
    assert not r["abstained"]
    assert r["citations"][0]["article_id"] == "DER-2026:4"
    assert r["citations"][0]["lang"] == "ar"  # prefers the question's language
    assert [t["node"] for t in r["trace"]] == [
        "detect_language",
        "rewrite_query",
        "retrieve",
        "select_context",
        "generate",
        "validate",
    ]


def test_end_to_end_english(sanad):
    r = sanad.ask("What is the maximum fee for settling a loan early?")
    assert r["citations"][0]["article_id"] == "DCFR-2026:5"
    assert "1%" in r["answer"]


def test_hallucinated_citations_are_removed(index):
    s = Sanad(index, ScriptedLLM("Overtime is paid at plus 50% [S1]. Also something [S9]."), use_langgraph=False)
    r = s.ask("How much is overtime paid at night?")
    assert "[S9]" not in r["answer"] and "[S1]" in r["answer"]
    assert [c["marker"] for c in r["citations"]] == ["S1"]


def test_abstains_without_citations(index):
    s = Sanad(index, ScriptedLLM("I think it is probably fine."), use_langgraph=False)
    r = s.ask("How much is overtime paid at night?")
    assert r["abstained"] and r["answer"] == ABSTAIN["en"]


def test_abstains_on_not_found_in_question_language(index):
    s = Sanad(index, ScriptedLLM("NOT_FOUND"), use_langgraph=False)
    r = s.ask("ما هي عاصمة فرنسا؟")
    assert r["abstained"] and r["answer"] == ABSTAIN["ar"]


def test_rewritten_query_is_used(index):
    s = Sanad(index, ScriptedLLM("x [S1]", rewrite="maximum early settlement fee"), use_langgraph=False)
    r = s.ask("كم رسوم السداد المبكر؟")
    assert r["rewritten_queries"] == ["maximum early settlement fee"]


def test_rewrite_failure_does_not_break_request(index):
    class Flaky(ScriptedLLM):
        def generate(self, system, prompt, temperature=0.1):
            if "REWRITE" in system:
                raise TimeoutError
            return "Answer [S1]"

    r = Sanad(index, Flaky(""), use_langgraph=False).ask("How much is overtime paid at night?")
    assert not r["abstained"]
    assert r["trace"][1]["note"].startswith("skipped")


def test_empty_question(sanad):
    with pytest.raises(ValueError):
        sanad.ask("   ")


def test_langgraph_engine_matches_sequential(index):
    pytest.importorskip("langgraph")
    from sanad.llm import MockLLM

    q = "هل يستحق العامل مكافأة نهاية الخدمة إذا أُنهي عقده خلال فترة التجربة؟"
    lg = Sanad(index, MockLLM(), use_langgraph=True)
    seq = Sanad(index, MockLLM(), use_langgraph=False)
    assert lg.app is not None
    a, b = lg.ask(q), seq.ask(q)
    assert a["answer"] == b["answer"] and a["citations"] == b["citations"]
    assert a["engine"] == "langgraph"

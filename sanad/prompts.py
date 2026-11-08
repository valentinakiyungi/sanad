"""Prompt templates. Kept in one place so they can be reviewed and versioned."""

LANG_NAME = {"ar": "Arabic", "en": "English"}

ABSTAIN = {
    "en": (
        "I couldn't find anything about that in the regulations I cover. Try asking about employment or consumer finance rules."
    ),
    "ar": "لم أجد ما يتعلق بهذا السؤال في اللوائح المتوفرة. جرّب السؤال عن أحكام العمل أو التمويل الاستهلاكي.",
}

REWRITE_SYSTEM = (
    "REWRITE. You translate legal search queries between Arabic and English. "
    "Return ONLY the translated query on one line: no quotes, no explanation. "
    "Use formal legal terminology (Modern Standard Arabic for Arabic)."
)


def rewrite_prompt(question: str, target_lang: str) -> str:
    return f"Translate this question into {LANG_NAME[target_lang]} as a search query:\n{question}"


ANSWER_SYSTEM = """You are Sanad, an assistant that answers questions about regulations.

Rules:
1. Use ONLY the numbered sources provided. Never use outside knowledge.
2. Put a citation marker like [S1] after every sentence that states a rule, number or condition.
3. If a source refers to another article that is also provided, combine them into one complete answer.
4. Answer in {language}, even if the sources are in another language.
5. Be concise: 1-4 sentences. Quote numbers exactly as written in the sources.
6. If the sources do not contain the answer, reply with exactly: NOT_FOUND"""


def answer_prompt(question: str, sources: list[dict]) -> str:
    blocks = [f"[{s['marker']}] {s['label']} — {s['title']}\n{s['text']}" for s in sources]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"

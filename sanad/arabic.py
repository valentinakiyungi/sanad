"""Arabic + English text normalisation, tokenisation and light stemming.

Arabic text is noisy for retrieval: the same word can appear with or without
diacritics, with different alef/hamza forms, with tatweel stretching, or with
attached prefixes (و، ال، ب، ل). Normalising these is the single biggest win
for Arabic BM25, so it lives in one well-tested module.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat (fatha, damma, kasra, tanwin, shadda, sukun, ...) and superscript alef.
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ARABIC_LETTER = re.compile(r"[؀-ۿ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[ء-ي0-9]+|[a-z0-9]+")

_CHAR_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
        # Arabic-Indic and Eastern Arabic-Indic digits -> ASCII
        **{chr(0x0660 + i): str(i) for i in range(10)},
        **{chr(0x06F0 + i): str(i) for i in range(10)},
        "٪": "%",
        "،": ",",
        "؛": ";",
        "؟": "?",
    }
)

_AR_STOPWORD_TEXT = """
في من على الى عن مع او ان أن إن ما ماذا هي هو هل كم كيف متى اين التي الذي الذين
هذه هذا ذلك تلك كل اي بين قبل بعد عند خلال لا لم لن قد ثم و ف ب ل يكون تكون
"""

_EN_STOPWORD_TEXT = """
a an the and or of to in on at for by with from is are was were be been being it its
this that these those what which who whom how when where why do does did can could
shall should will would may might must i me my we our you your he she they them their
there any all not no if than then as into about after before during under over per
"""

EN_STOPWORDS = frozenset(_EN_STOPWORD_TEXT.split())

_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ل", "ف")
_AR_SUFFIXES = ("ات", "ون", "ين", "ان", "ها", "هم", "يه", "ه", "ي")


def normalize(text: str) -> str:
    """Normalise Arabic orthography and case. Safe to apply to mixed text."""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_CHAR_MAP)
    return text.lower()


AR_STOPWORDS = frozenset(normalize(w) for w in _AR_STOPWORD_TEXT.split())


def detect_language(text: str) -> str:
    """Return 'ar' or 'en' by script majority (ties and empty text -> 'en')."""
    ar = len(_ARABIC_LETTER.findall(text))
    en = len(_LATIN_LETTER.findall(text))
    return "ar" if ar > en else "en"


def stem_ar(token: str) -> str:
    """Light Arabic stemmer: strip one common prefix and one suffix.

    Deliberately conservative (keeps >= 3 letters) — aggressive root
    extraction hurts precision on legal text more than it helps recall.
    """
    for p in _AR_PREFIXES:
        # Single letters (و ب ل ف) are often part of the word itself (فترة "period"),
        # so they are only stripped from longer words.
        min_left = 3 if len(p) > 1 else 4
        if token.startswith(p) and len(token) - len(p) >= min_left:
            token = token[len(p) :]
            break
    for s in _AR_SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[: -len(s)]
            break
    return token


def stem_en(token: str) -> str:
    """Tiny English stemmer good enough for BM25 on regulations."""
    for suf, min_len in (("ies", 5), ("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if token.endswith(suf) and len(token) >= min_len and not token.endswith("ss"):
            return token[: -len(suf)] + ("y" if suf == "ies" else "")
    return token


def tokenize(text: str, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    """Normalise, split and (optionally) stem mixed Arabic/English text."""
    out: list[str] = []
    for tok in _TOKEN.findall(normalize(text)):
        is_ar = bool(_ARABIC_LETTER.match(tok))
        if drop_stopwords and tok in (AR_STOPWORDS if is_ar else EN_STOPWORDS):
            continue
        if stem and not tok.isdigit():
            tok = stem_ar(tok) if is_ar else stem_en(tok)
        out.append(tok)
    return out

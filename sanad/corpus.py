"""Load regulations and split them into article-level chunks.

Legal text has natural retrieval units: articles. Chunking on article
boundaries (instead of fixed token windows) keeps every citation exact and
lets the knowledge graph link "Article (8)" mentions to the right node.

File naming convention: ``<law-id>.<lang>.<ext>`` e.g. ``der-2026.ar.md``.
Supported extensions: .md, .txt and .pdf (PDF needs ``pypdf``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .arabic import detect_language

_DIGITS = str.maketrans({**{chr(0x0660 + i): str(i) for i in range(10)}, **{chr(0x06F0 + i): str(i) for i in range(10)}})

_ARTICLE_HEADER = re.compile(
    r"^\s*#{0,6}\s*(?:Article|ART\.?|المادة|مادة)\s*\(?\s*(\d+)\s*\)?\s*[-–—:.]?\s*(.*)$",
    re.IGNORECASE,
)
_TITLE = re.compile(r"^\s*#\s+(.+)$")
# "Article (8)", "Articles (3) and (4)", "المادة (8)", "للمادة (3)", "المادتين (3) و(4)"
_REF = re.compile(
    r"(?:\bArticles?\b|(?:ال|لل|بال|وال)?ماد(?:ة|تين|تان)|المواد)\s*"
    r"((?:\(?\s*\d+\s*\)?\s*(?:,|،|and|or|و|أو)?\s*)+)",
    re.IGNORECASE,
)
_DEF_EN = re.compile(r'"([^"\n]{2,60})"\s*:')
_DEF_AR = re.compile(r"«([^»\n]{2,60})»\s*:")


@dataclass
class Chunk:
    law_id: str
    law_title: str
    article: int
    title: str
    lang: str
    text: str
    references: list[int] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return f"{self.law_id}:{self.article}:{self.lang}"

    @property
    def article_id(self) -> str:
        """Language-independent id: the same article in Arabic and English share it."""
        return f"{self.law_id}:{self.article}"

    def label(self) -> str:
        word = "المادة" if self.lang == "ar" else "Art."
        return f"{self.law_id}, {word} {self.article}"

    def to_dict(self) -> dict:
        return asdict(self)


def find_references(text: str) -> list[int]:
    """Article numbers mentioned in a body of text, in order of first mention."""
    seen: list[int] = []
    for span in _REF.findall(text.translate(_DIGITS)):
        for n in re.findall(r"\d+", span):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


def find_definitions(text: str) -> list[str]:
    return [t.strip() for t in _DEF_EN.findall(text) + _DEF_AR.findall(text)]


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install pypdf to ingest PDF files: pip install pypdf") from exc
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def parse_document(text: str, law_id: str, lang: str | None = None) -> list[Chunk]:
    """Split one regulation into article chunks."""
    lang = lang or detect_language(text)
    law_title = law_id
    chunks: list[Chunk] = []
    current: dict | None = None
    body: list[str] = []

    def flush() -> None:
        if current is None:
            return
        content = "\n".join(body).strip()
        if not content:
            return
        refs = [r for r in find_references(content) if r != current["article"]]
        chunks.append(
            Chunk(
                law_id=law_id,
                law_title=law_title,
                article=current["article"],
                title=current["title"],
                lang=lang,
                text=content,
                references=refs,
                defines=find_definitions(content),
            )
        )

    for line in text.splitlines():
        header = _ARTICLE_HEADER.match(line.translate(_DIGITS))
        if header:
            flush()
            current = {"article": int(header.group(1)), "title": header.group(2).strip()}
            body = []
            continue
        title = _TITLE.match(line)
        if title and current is None:
            law_title = title.group(1).strip()
            continue
        if current is not None:
            body.append(line)
    flush()
    return chunks


def load_corpus(directory: str | Path) -> list[Chunk]:
    """Load every ``<law>.<lang>.<ext>`` file in a directory."""
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        parts = path.stem.split(".")
        law_id = parts[0].upper()
        lang = parts[1] if len(parts) > 1 and parts[1] in {"ar", "en"} else None
        chunks.extend(parse_document(_read_text(path), law_id, lang))
    if not chunks:
        raise ValueError(f"No articles found in {directory}. Expected files like 'law-id.ar.md'.")
    return chunks

"""Settings from environment variables, with offline-friendly defaults.

With GEMINI_API_KEY set, Sanad uses Gemini for embeddings, query rewriting and
answers. Without it, everything falls back to the offline hash embedder and
the mock LLM, so the project always runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .embeddings import Embedder, GeminiEmbedder, HashEmbedder
from .llm import LLM, GeminiClient, MockLLM


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    llm: str
    embedder: str
    embed_model: str
    embed_dim: int
    index_dir: str
    corpus_dir: str
    mode: str

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()
        default = "gemini" if os.environ.get("GEMINI_API_KEY") else "offline"
        return cls(
            llm=os.environ.get("SANAD_LLM", "gemini" if default == "gemini" else "mock"),
            embedder=os.environ.get("SANAD_EMBEDDER", "gemini" if default == "gemini" else "hash"),
            embed_model=os.environ.get("SANAD_EMBED_MODEL", "gemini-embedding-001"),
            embed_dim=int(os.environ.get("SANAD_EMBED_DIM", "768")),
            index_dir=os.environ.get("SANAD_INDEX_DIR", ".index"),
            corpus_dir=os.environ.get("SANAD_CORPUS_DIR", "data/demo"),
            mode=os.environ.get("SANAD_RETRIEVAL_MODE", "graph"),
        )

    def make_llm(self) -> LLM:
        return GeminiClient() if self.llm == "gemini" else MockLLM()

    def make_embedder(self) -> Embedder:
        if self.embedder == "gemini":
            return GeminiEmbedder(GeminiClient(), self.embed_model, self.embed_dim)
        return HashEmbedder()

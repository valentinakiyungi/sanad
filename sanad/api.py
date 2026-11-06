"""FastAPI service: `uvicorn sanad.api:app --reload` then open http://localhost:8000"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .cli import _load_or_build
from .config import Settings
from .llm import GeminiError
from .pipeline import Sanad
from .retrieval import MODES

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Sanad", version=__version__, description="Bilingual Arabic-English GraphRAG over regulations")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000, examples=["ما هي مدة فترة التجربة؟"])
    mode: str | None = Field(default=None, description=f"one of {MODES}")


@lru_cache(maxsize=len(MODES) + 1)
def get_engine(mode: str | None = None) -> Sanad:
    settings = Settings.from_env()
    return Sanad(_load_or_build(settings), settings.make_llm(), mode=mode or settings.mode)


@app.get("/api/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "version": __version__,
        "chunks": len(engine.index.chunks),
        "embedder": engine.index.embedder.name,
        "model": engine.llm.name,
        "engine": "langgraph" if engine.app is not None else "sequential",
        "graph": engine.index.graph.stats(),
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    if req.mode is not None and req.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    try:
        return get_engine(req.mode).ask(req.question)
    except GeminiError as e:
        raise HTTPException(502, str(e)) from e


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")

"""FastAPI service: `uvicorn sanad.api:app --reload` then open http://localhost:8000"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .cli import _load_or_build
from .config import Settings
from .llm import GeminiError
from .pipeline import Sanad
from .retrieval import MODES

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("sanad")

# A public demo spends the owner's Gemini quota, so limit questions per visitor.
RATE_LIMIT = int(os.environ.get("SANAD_RATE_LIMIT_PER_MIN", "10"))
_requests: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Build or load the index at startup, so the first visitor doesn't wait for embeddings.
    if os.environ.get("SANAD_WARMUP", "1") == "1":
        engine = get_engine()
        log.warning("Sanad ready: %d chunks, %s, %s", len(engine.index.chunks), engine.index.embedder.name, engine.llm.name)
    yield


app = FastAPI(
    title="Sanad",
    version=__version__,
    description="Bilingual Arabic-English GraphRAG over regulations",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000, examples=["ما هي مدة فترة التجربة؟"])
    mode: str | None = Field(default=None, description=f"one of {MODES}")


@lru_cache(maxsize=len(MODES) + 1)
def get_engine(mode: str | None = None) -> Sanad:
    settings = Settings.from_env()
    return Sanad(_load_or_build(settings), settings.make_llm(), mode=mode or settings.mode)


def _check_rate(request: Request) -> None:
    if RATE_LIMIT <= 0:
        return
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else "?")
    now, window = time.monotonic(), _requests[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, f"Demo limit: {RATE_LIMIT} questions per minute. Please wait a moment.")
    window.append(now)


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
def ask(req: AskRequest, request: Request) -> dict:
    if req.mode is not None and req.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    _check_rate(request)
    try:
        return get_engine(req.mode).ask(req.question)
    except GeminiError as e:
        log.error("Gemini error: %s", e)
        busy = "429" in str(e)
        raise HTTPException(503 if busy else 502, "The model is busy, try again shortly." if busy else str(e)) from e


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")

"""LLM clients: Gemini over plain REST (stdlib only) and an offline mock.

The Gemini client uses urllib so the core package has no HTTP dependency and
runs in restricted environments. It retries on 429/5xx with backoff.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Protocol

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
# Pinned stable model (fast, cheap, good at short cited answers). Override with SANAD_LLM_MODEL,
# e.g. gemini-3.8-flash for the strongest Flash model.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class LLM(Protocol):
    name: str

    def generate(self, system: str, prompt: str, temperature: float = 0.1) -> str: ...


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 60, retries: int = 4):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey")
        self.model = model or os.environ.get("SANAD_LLM_MODEL", DEFAULT_MODEL)
        self.timeout, self.retries = timeout, retries
        self.name = f"gemini:{self.model}"

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{API_ROOT}/{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise GeminiError(f"Gemini API {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                # A slow response raises TimeoutError while reading, not URLError,
                # so both are retried with backoff.
                if attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                reason = getattr(e, "reason", None) or "request timed out"
                raise GeminiError(f"Cannot reach Gemini API: {reason}") from e
        raise GeminiError("unreachable")

    def generate(self, system: str, prompt: str, temperature: float = 0.1) -> str:
        data = self._post(
            f"models/{self.model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            },
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as e:
            raise GeminiError(f"Unexpected Gemini response: {json.dumps(data)[:300]}") from e
        return "".join(p.get("text", "") for p in parts).strip()

    def embed(self, texts: list[str], model: str, task_type: str, dim: int) -> list[list[float]]:
        data = self._post(
            f"models/{model}:batchEmbedContents",
            {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task_type,
                        "outputDimensionality": dim,
                    }
                    for t in texts
                ]
            },
        )
        return [e["values"] for e in data["embeddings"]]


class MockLLM:
    """Deterministic offline stand-in: answers extractively from the first source.

    Lets the whole pipeline, API and tests run without a key. Query rewriting
    returns nothing, so retrieval relies on the graph's language twins.
    """

    name = "mock"
    _SOURCE = re.compile(r"\[S(\d+)\][^\n]*\n(.+?)(?=\n\[S\d+\]|\Z)", re.S)

    def generate(self, system: str, prompt: str, temperature: float = 0.1) -> str:
        if "REWRITE" in system:
            return ""
        sources = self._SOURCE.findall(prompt)
        if not sources:
            return "NOT_FOUND"
        num, text = sources[0]
        sentences = re.split(r"(?<=[.!؟?])\s+", text.strip())
        body = " ".join(s for s in sentences[:2] if s.strip())
        return f"{body} [S{num}]"

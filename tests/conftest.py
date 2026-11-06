import os
from pathlib import Path

import pytest

from sanad.corpus import load_corpus
from sanad.embeddings import HashEmbedder
from sanad.llm import MockLLM
from sanad.pipeline import Sanad
from sanad.retrieval import Index

ROOT = Path(__file__).resolve().parent.parent

# Tests always run offline, even if a developer has a key in their shell.
os.environ.pop("GEMINI_API_KEY", None)


@pytest.fixture(scope="session")
def chunks():
    return load_corpus(ROOT / "data" / "demo")


@pytest.fixture(scope="session")
def index(chunks):
    return Index.build(chunks, HashEmbedder())


@pytest.fixture(scope="session")
def sanad(index):
    return Sanad(index, MockLLM(), use_langgraph=False)

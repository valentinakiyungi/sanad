import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv("SANAD_INDEX_DIR", str(tmp_path_factory.mktemp("index")))
    monkeypatch_module.setenv("SANAD_LLM", "mock")
    monkeypatch_module.setenv("SANAD_EMBEDDER", "hash")
    from sanad import api

    api.get_engine.cache_clear()
    return TestClient(api.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["chunks"] == 32


def test_ask(client):
    r = client.post("/api/ask", json={"question": "ما هي النسبة القصوى لعبء الدين؟"})
    assert r.status_code == 200
    assert r.json()["citations"][0]["article_id"] == "DCFR-2026:3"


def test_ask_validation(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "hello", "mode": "nope"}).status_code == 422


def test_rate_limit(client, monkeypatch):
    from sanad import api

    monkeypatch.setattr(api, "RATE_LIMIT", 2)
    api._requests.clear()
    codes = [client.post("/api/ask", json={"question": "overtime pay"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    api._requests.clear()


def test_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "Sanad" in r.text

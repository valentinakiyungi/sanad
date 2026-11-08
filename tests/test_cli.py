from sanad.cli import main


def test_check_offline(monkeypatch, capsys):
    monkeypatch.setenv("SANAD_LLM", "mock")
    monkeypatch.setenv("SANAD_EMBEDDER", "hash")
    main(["check"])
    assert "OFFLINE" in capsys.readouterr().out


def test_ask_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SANAD_LLM", "mock")
    monkeypatch.setenv("SANAD_EMBEDDER", "hash")
    monkeypatch.setenv("SANAD_INDEX_DIR", str(tmp_path))
    main(["ask", "كم رسوم السداد المبكر للقرض؟"])
    assert "DCFR-2026" in capsys.readouterr().out

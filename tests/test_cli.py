import pytest

from research_forest import cli
from research_forest.models import Paper
from research_forest.store import Store


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "cli.forest.db")
    store = Store(path)
    p = store.create_project("efficient attention")
    store.add_papers(p, [Paper(id="A", title="FlashAttention paper")])
    store.update_papers(p, ["A"], status="included")
    store.reindex(p)
    return path


def run(capsys, *argv):
    cli.main(list(argv))
    return capsys.readouterr().out


def test_list_search_undo(db, capsys):
    assert "efficient attention" in run(capsys, "--db", db, "list")
    assert "p:A" in run(capsys, "--db", db, "search", "1", "flash")
    assert "nothing to undo" in run(capsys, "--db", db, "undo", "1")


def test_delete_asks_for_confirmation(db, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert "cancelled" in run(capsys, "--db", db, "delete", "1")
    assert "efficient attention" in run(capsys, "--db", db, "list")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert "deleted landscape 1" in run(capsys, "--db", db, "delete", "1")
    assert run(capsys, "--db", db, "list") == ""
    with pytest.raises(SystemExit, match="no landscape 1"):
        cli.main(["--db", db, "delete", "1", "--yes"])


def test_argument_validation(db):
    with pytest.raises(SystemExit):
        cli.main(["--db", db, "--backend", "bogus", "list"])
    with pytest.raises(SystemExit):
        cli.main(["--db", db])


def test_serve_prints_tokenised_url(db, monkeypatch, capsys):
    import uvicorn
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw, token=app.state.token))
    cli.main(["--db", db, "serve", "--host", "0.0.0.0", "--port", "9001"])
    err = capsys.readouterr().err
    assert f"http://127.0.0.1:9001/?token={seen['token']}" in err and "warning" in err
    assert seen["host"] == "0.0.0.0" and seen["port"] == 9001

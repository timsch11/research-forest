from dataclasses import fields
from pathlib import Path

import pytest

from research_forest import cli
from research_forest.config import CONFIG_KEYS, SECRETS, ConfigError, Settings, read_config

ROOT = Path(__file__).resolve().parents[1]
SECRET_VARS = [v for vs in SECRETS.values() for v in vs]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in SECRET_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_defaults_without_files():
    assert Settings.load() == Settings()


def test_shipped_config_is_valid_and_matches_defaults():
    values = read_config(ROOT / "research-forest.toml")
    defaults = Settings()
    assert values and all(getattr(defaults, k) == v for k, v in values.items())
    documented = {field for field, _ in CONFIG_KEYS.values()} - {"model"}  # model is commented out (no default)
    assert documented == set(values)


def test_every_non_secret_setting_is_configurable():
    configurable = {field for field, _ in CONFIG_KEYS.values()}
    assert configurable | set(SECRETS) == {f.name for f in fields(Settings)}


def test_config_file_is_loaded_from_cwd(tmp_path):
    (tmp_path / "research-forest.toml").write_text(
        '[llm]\nbackend = "claude-code"\nmodel = "opus"\ntimeout = 60\n'
        '[pipeline]\ntarget_papers = 80\n[storage]\ndb_path = "x.db"\n')
    s = Settings.load()
    assert (s.backend, s.model, s.llm_timeout, s.target_papers, s.db_path) == ("claude-code", "opus", 60.0, 80, "x.db")
    assert isinstance(s.llm_timeout, float)


def test_explicit_config_path_and_cli_overrides(tmp_path):
    cfg = tmp_path / "custom.toml"
    cfg.write_text('[llm]\nmodel = "from-file"\n[storage]\ndb_path = "file.db"\n')
    s = Settings.load(cfg, model="from-cli", db_path=None)
    assert s.model == "from-cli" and s.db_path == "file.db"  # None overrides are ignored
    with pytest.raises(ConfigError, match="not found"):
        Settings.load(tmp_path / "missing.toml")


def test_secrets_from_dotenv_and_terminal_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# comment\nexport OPENAI_API_KEY='from-file'\nS2_API_KEY=s2-file  # trailing comment\n"
        'RF_API_TOKEN="tok"\nnot a pair\n')
    monkeypatch.setenv("OPENAI_API_KEY", "from-terminal")
    s = Settings.load()
    assert (s.openai_api_key, s.s2_api_key, s.api_token) == ("from-terminal", "s2-file", "tok")


def test_non_secret_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("RF_TARGET_PAPERS", "999")
    monkeypatch.setenv("RF_BACKEND", "codex")
    defaults = Settings()
    assert Settings.load().target_papers == defaults.target_papers and Settings.load().backend == defaults.backend


@pytest.mark.parametrize("toml, message", [
    ('[llm]\napi_key = "sk-x"\n', "secrets belong in .env"),
    ('[llm]\nOPENAI_API_KEY = "sk-x"\n', "secrets belong in .env"),
    ('[server]\napi_token = "t"\n', "secrets belong in .env"),
    ('[llm]\nbakend = "openai"\n', "unknown setting \\[llm\\] bakend"),
    ('[pipeline]\ntarget_papers = "many"\n', "must be int"),
    ('[pipeline]\ntarget_papers = 0\n', "between 5 and 400"),
    ('[pipeline]\ntarget_papers = 9999\n', "between 5 and 400"),
    ('[pipeline]\nfollowup_papers = -1\n', "must not be negative"),
    ('[semantic_scholar]\nrequests_per_second = -1.0\n', "must not be negative"),
    ('[pipeline]\ntarget_papers = true\n', "must be int"),
    ('[fulltext]\nenabled = "yes"\n', "must be bool"),
    ('[llm]\nbackend = "gemini"\n', "backend must be one of"),
    ('[llm]\neffort = "extreme"\n', "effort must be one of"),
    ('backend = "openai"\n', "must be a \\[section\\]"),
    ('[llm\n', "invalid TOML"),
])
def test_invalid_config_is_rejected(tmp_path, toml, message):
    (tmp_path / "research-forest.toml").write_text(toml)
    with pytest.raises(ConfigError, match=message):
        Settings.load()


def test_cli_reports_config_errors(tmp_path, capsys):
    (tmp_path / "bad.toml").write_text('[llm]\napi_key = "sk"\n')
    with pytest.raises(SystemExit, match="secrets belong in .env"):
        cli.main(["--config", str(tmp_path / "bad.toml"), "list"])


def test_cli_uses_config_file(tmp_path, capsys):
    (tmp_path / "research-forest.toml").write_text(f'[storage]\ndb_path = "{(tmp_path / "cfg.db").as_posix()}"\n')
    cli.main(["list"])
    assert (tmp_path / "cfg.db").exists()



# ------------------------------------------------- risky settings from the config file


def risky_config(tmp_path, body):
    (tmp_path / "research-forest.toml").write_text(body)


@pytest.mark.parametrize("body, flag", [
    ('[llm]\nbackend = "openai"\nbase_url = "https://evil.example/v1"\n', ["--base-url", "https://evil.example/v1"]),
    ('[fulltext]\nallow_private = true\n', ["--allow-private"]),
])
def test_risky_config_needs_consent_or_a_cli_flag(tmp_path, monkeypatch, capsys, body, flag):
    risky_config(tmp_path, body)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit, match="refusing to continue"):  # non-interactive: no silent consent
        cli.main(["list"])
    assert "warning:" in capsys.readouterr().err
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(SystemExit, match="stopped"):
        cli.main(["list"])
    monkeypatch.setattr("builtins.input", lambda _: "y")
    cli.main(["list"])  # consent given
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("must not ask when the CLI flag is set"))
    cli.main([*flag, "list"])


@pytest.mark.parametrize("body, env", [
    ('[llm]\nbackend = "openai"\nbase_url = "http://localhost:11434/v1"\n', {"OPENAI_API_KEY": "sk-test"}),
    ('[llm]\nbackend = "openai"\nbase_url = "https://evil.example/v1"\n', {}),  # no key to leak
    ('[llm]\nbackend = "claude-code"\nbase_url = "https://evil.example/v1"\n', {"OPENAI_API_KEY": "sk-test"}),
    ('[llm]\nbackend = "openai"\nbase_url = "https://api.openai.com/v1"\n', {"OPENAI_API_KEY": "sk-test"}),
])
def test_harmless_configs_are_not_questioned(tmp_path, monkeypatch, body, env):
    from research_forest.config import untrusted_settings
    risky_config(tmp_path, body)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert untrusted_settings(Settings.load(), None, set()) == []


def test_build_rejects_out_of_range_paper_counts(capsys):
    for n in ("0", "4", "401", "-3"):
        with pytest.raises(SystemExit):
            cli.main(["build", "x", "-n", n])
    assert "between 5 and 400" in capsys.readouterr().err

from backend.app.config import get_settings


def test_settings_loads():
    settings = get_settings()
    assert settings.APP_NAME == "HHGoa-RAG"
    assert settings.APP_VERSION == "0.1.0"
    assert settings.PORT == 8000


def test_settings_defaults(monkeypatch):
    from backend.app.config import Settings
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.DEBUG is False
    assert settings.LOG_LEVEL == "INFO"
    assert settings.LLM_PROVIDER == ""  # not locked to any provider
    assert settings.EMBEDDING_MODEL == ""  # not locked to any model


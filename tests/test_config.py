import pytest
from pydantic import SecretStr

from archivistbot.config import Settings


def test_database_password_has_no_insecure_default() -> None:
    settings = Settings(
        _env_file=None,
        database_url=None,
        postgres_password=SecretStr(""),
    )

    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        _ = settings.sqlalchemy_url


def test_importing_bot_helpers_does_not_require_database_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from archivistbot.bot import extract_command_query, is_bot_authored_media

    assert extract_command_query("/find cat") == "cat"
    assert callable(is_bot_authored_media)

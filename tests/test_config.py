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

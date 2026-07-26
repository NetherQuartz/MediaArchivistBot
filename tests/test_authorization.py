from types import SimpleNamespace

import pytest

from archivistbot.search import get_allowed_chat_ids


class FakeBot:
    bot_id = 99

    async def get_chat_member(self, chat_id: int, user_id: int):
        if chat_id == 5:
            raise RuntimeError("Telegram unavailable")
        if user_id == self.bot_id:
            return SimpleNamespace(status="administrator")
        statuses = {
            1: SimpleNamespace(status="member"),
            2: SimpleNamespace(status="left"),
            3: SimpleNamespace(status="restricted", is_member=True),
            4: SimpleNamespace(status="restricted", is_member=False),
        }
        return statuses[chat_id]


@pytest.mark.asyncio
async def test_membership_checks_fail_closed() -> None:
    allowed = await get_allowed_chat_ids(
        FakeBot(),
        user_id=42,
        chat_ids=[1, 2, 3, 4, 5],
        concurrency=2,
    )

    assert allowed == [1, 3]

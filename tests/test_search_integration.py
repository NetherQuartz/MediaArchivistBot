import os

import pytest

from archivistbot.config import Settings
from archivistbot.database import session_scope
from archivistbot.models import Chat, Media, Message, ProcessingStatus
from archivistbot.search import SearchService

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_TESTS") != "1",
    reason="Set RUN_DB_TESTS=1 and run against a migrated test database",
)


class FakeEmbeddings:
    async def embed_query(self, query: str) -> list[float]:
        return [1.0] + [0.0] * 1023


@pytest.mark.asyncio
async def test_search_filters_chat_before_ranking() -> None:
    allowed_chat_id = -9_100_001
    denied_chat_id = -9_100_002
    with session_scope() as session:
        session.add_all(
            [
                Chat(
                    chat_id=allowed_chat_id,
                    type="supergroup",
                    active=True,
                    bot_status="administrator",
                ),
                Chat(
                    chat_id=denied_chat_id,
                    type="supergroup",
                    active=True,
                    bot_status="administrator",
                ),
            ]
        )
        session.flush()
        allowed_message = Message(
            chat_id=allowed_chat_id,
            sender_id=1,
            message_id=101,
            caption="capybara",
        )
        denied_message = Message(
            chat_id=denied_chat_id,
            sender_id=2,
            message_id=202,
            caption="capybara",
        )
        session.add_all([allowed_message, denied_message])
        session.flush()
        session.add_all(
            [
                Media(
                    message_uuid=allowed_message.message_uuid,
                    file_id="allowed",
                    file_unique_id="allowed",
                    media_type="image",
                    status=ProcessingStatus.READY,
                    search_text="a capybara sits on a person's head",
                    embedding=[0.9, 0.1] + [0.0] * 1022,
                    embedding_model="qwen3-embedding:0.6b",
                ),
                Media(
                    message_uuid=denied_message.message_uuid,
                    file_id="denied",
                    file_unique_id="denied",
                    media_type="image",
                    status=ProcessingStatus.READY,
                    search_text="a capybara sits on a person's head",
                    embedding=[1.0] + [0.0] * 1023,
                    embedding_model="qwen3-embedding:0.6b",
                ),
            ]
        )

    service = SearchService(
        embedding_service=FakeEmbeddings(),
        settings=Settings(search_candidates=10),
    )
    with session_scope() as session:
        results = await service.search(
            session,
            "capybara on a person's head",
            [allowed_chat_id],
        )

    try:
        assert results
        assert {result.chat_id for result in results} == {allowed_chat_id}
    finally:
        with session_scope() as session:
            for chat_id in (allowed_chat_id, denied_chat_id):
                chat = session.get(Chat, chat_id)
                if chat is not None:
                    session.delete(chat)

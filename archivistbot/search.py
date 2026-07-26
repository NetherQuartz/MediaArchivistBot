import asyncio
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlmodel import Session
from telebot.async_telebot import AsyncTeleBot

from .config import Settings, get_settings
from .llm_api import EmbeddingService
from .models import Chat, Media, Message, ProcessingStatus

logger = logging.getLogger(__name__)
RRF_K = 60


@dataclass(frozen=True)
class SearchResult:
    media_uuid: uuid.UUID
    chat_id: int
    message_id: int
    file_id: str
    file_unique_id: str | None
    media_type: str
    mime_type: str | None
    title: str
    chat_title: str | None
    score: float
    cosine_distance: float | None


async def get_allowed_chat_ids(
    bot: AsyncTeleBot,
    user_id: int,
    chat_ids: list[int],
    concurrency: int,
    bot_user_id: int | None = None,
) -> list[int]:
    semaphore = asyncio.Semaphore(concurrency)
    verified_bot_id = bot_user_id or bot.bot_id

    async def check(chat_id: int) -> int | None:
        async with semaphore:
            try:
                bot_member = await bot.get_chat_member(chat_id, verified_bot_id)
                if str(bot_member.status) != "administrator":
                    return None
                member = await bot.get_chat_member(chat_id, user_id)
            # Authorization must fail closed for every Telegram/network failure.
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "Membership check failed for user=%s chat=%s: %s",
                    user_id,
                    chat_id,
                    error,
                )
                return None

            status = str(member.status)
            if status in {"creator", "administrator", "member"}:
                return chat_id
            if status == "restricted" and bool(getattr(member, "is_member", False)):
                return chat_id
            return None

    results = await asyncio.gather(*(check(chat_id) for chat_id in chat_ids))
    return [chat_id for chat_id in results if chat_id is not None]


class SearchService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or EmbeddingService(self.settings)

    async def search(
        self,
        session: Session,
        query: str,
        allowed_chat_ids: list[int],
    ) -> list[SearchResult]:
        cleaned_query = " ".join(query.split())
        if not cleaned_query or not allowed_chat_ids:
            return []

        query_embedding = await self.embedding_service.embed_query(cleaned_query)
        distance = Media.embedding.cosine_distance(query_embedding).label(
            "cosine_distance"
        )
        semantic_rows = session.exec(
            select(Media.media_uuid, distance)
            .join(Message, Message.message_uuid == Media.message_uuid)
            .where(
                Message.chat_id.in_(allowed_chat_ids),
                Media.status == ProcessingStatus.READY,
                Media.embedding.is_not(None),
                Media.embedding_model == self.settings.embedding_model,
            )
            .order_by(distance)
            .limit(self.settings.search_candidates)
        ).all()

        ts_query = func.websearch_to_tsquery("simple", cleaned_query)
        text_rank = func.ts_rank_cd(Media.search_vector, ts_query)
        trigram_rank = func.similarity(Media.search_text, cleaned_query)
        lexical_score = (text_rank + trigram_rank).label("lexical_score")
        lexical_rows = session.exec(
            select(Media.media_uuid, lexical_score)
            .join(Message, Message.message_uuid == Media.message_uuid)
            .where(
                Message.chat_id.in_(allowed_chat_ids),
                Media.status == ProcessingStatus.READY,
                Media.search_text.is_not(None),
                or_(
                    Media.search_vector.op("@@")(ts_query),
                    trigram_rank >= 0.12,
                ),
            )
            .order_by(lexical_score.desc())
            .limit(self.settings.search_candidates)
        ).all()

        distances = {
            media_uuid: float(row_distance)
            for media_uuid, row_distance in semantic_rows
        }
        scores: dict[uuid.UUID, float] = {}
        for rank, (media_uuid, row_distance) in enumerate(semantic_rows, start=1):
            if float(row_distance) <= self.settings.max_cosine_distance:
                scores[media_uuid] = scores.get(media_uuid, 0) + 1 / (RRF_K + rank)
        for rank, (media_uuid, _) in enumerate(lexical_rows, start=1):
            scores[media_uuid] = scores.get(media_uuid, 0) + 1 / (RRF_K + rank)

        if not scores:
            return []

        rows = session.exec(
            select(Media, Message, Chat.title)
            .join(Message, Message.message_uuid == Media.message_uuid)
            .join(Chat, Chat.chat_id == Message.chat_id)
            .where(Media.media_uuid.in_(list(scores)))
        ).all()
        by_id = {
            media.media_uuid: (media, message, chat_title)
            for media, message, chat_title in rows
        }

        results: list[SearchResult] = []
        seen_files: set[str] = set()
        for media_uuid, score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            pair = by_id.get(media_uuid)
            if pair is None:
                continue
            media, message, chat_title = pair
            dedupe_key = media.file_unique_id or str(media.media_uuid)
            if dedupe_key in seen_files:
                continue
            seen_files.add(dedupe_key)
            results.append(
                SearchResult(
                    media_uuid=media.media_uuid,
                    chat_id=message.chat_id,
                    message_id=message.message_id,
                    file_id=media.file_id,
                    file_unique_id=media.file_unique_id,
                    media_type=media.media_type,
                    mime_type=media.mime_type,
                    title=media_result_title(media),
                    chat_title=chat_title,
                    score=score,
                    cosine_distance=distances.get(media.media_uuid),
                )
            )
        return results


def media_result_title(media: Media, *, max_length: int = 64) -> str:
    description = media.description if isinstance(media.description, dict) else None
    if description:
        summary = description.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()[:max_length]

    if media.search_text:
        for line in media.search_text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if ":" in cleaned:
                _, _, rest = cleaned.partition(":")
                cleaned = rest.strip() or cleaned
            if cleaned:
                return cleaned[:max_length]

    return media.media_type.replace("_", " ").title()


def active_chat_ids(session: Session) -> list[int]:
    return list(
        session.exec(
            select(Chat.chat_id).where(
                Chat.active.is_(True),
                Chat.bot_status == "administrator",
            )
        ).scalars()
    )

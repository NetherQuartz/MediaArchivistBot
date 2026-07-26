import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from telebot import types
from telebot.apihelper import ApiTelegramException
from telebot.async_telebot import AsyncTeleBot

from .config import Settings, get_settings
from .database import session_scope
from .models import (
    Chat,
    ChatType,
    Media,
    MediaType,
    ProcessingStatus,
    utc_now,
)
from .models import (
    Message as StoredMessage,
)
from .search import (
    SearchResult,
    SearchService,
    active_chat_ids,
    get_allowed_chat_ids,
)
from .worker import IndexWorker

logger = logging.getLogger(__name__)

WELCOME_TEXT = """
I index new images, GIFs, and videos from groups where I am an administrator.

Send me a plain-text description in a private chat, or use /find followed by a
description. I will find matching memes and forward their original messages.
Private search only covers groups you currently belong to.

Inside a group, use /search or /find followed by a description. Group search
only uses media indexed from that group.

You can also search inline anywhere: type @bot_username and a description. Inline
search covers groups where your membership is confirmed.

Note: Telegram does not expose messages sent before the bot joined.
""".strip()

HELP_TEXT = """
1. Add the bot to a group as an administrator. No extra permissions are needed.
2. New photos, GIFs, videos, and image documents will be indexed.
3. Send a private query as plain text or: /find Stilgar as it was written
4. In a group, use: /search cat in deep snow
5. Or search inline: @bot_username cat in deep snow

The bot does not persist media files. Deleted messages and protected content
cannot be retrieved.
""".strip()

INLINE_MIN_QUERY_LENGTH = 2


@dataclass(frozen=True)
class MediaCandidate:
    file_id: str
    file_unique_id: str | None
    media_type: str
    mime_type: str | None
    file_size: int | None
    duration: int | None


class BotApplication:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.validate_runtime()
        self.bot = AsyncTeleBot(self.settings.tg_token.get_secret_value())
        self.worker = IndexWorker(self.bot, self.settings)
        self.search_service = SearchService(settings=self.settings)
        self.bot_user_id: int | None = None
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.bot.register_message_handler(
            self.start_command,
            commands=["start"],
            chat_types=["private"],
        )
        self.bot.register_message_handler(
            self.help_command,
            commands=["help"],
            chat_types=["private"],
        )
        self.bot.register_message_handler(
            self.private_search_command,
            commands=["search", "find"],
            chat_types=["private"],
            content_types=["text"],
        )
        self.bot.register_message_handler(
            self.index_media,
            chat_types=["group", "supergroup"],
            content_types=["photo", "video", "animation", "document"],
        )
        self.bot.register_message_handler(
            self.group_search,
            commands=["search", "find"],
            chat_types=["group", "supergroup"],
            content_types=["text"],
        )
        self.bot.register_message_handler(
            self.search,
            chat_types=["private"],
            content_types=["text"],
            func=lambda message: bool(message.text)
            and not message.text.lstrip().startswith("/"),
        )
        self.bot.register_inline_handler(
            self.inline_search,
            func=lambda _query: True,
        )
        self.bot.register_my_chat_member_handler(self.chat_membership)

    async def run(self) -> None:
        me = await self.bot.get_me()
        self.bot_user_id = me.id
        await self._reconcile_known_chats()
        await self.worker.start()
        try:
            await self.bot.infinity_polling(
                skip_pending=False,
                timeout=30,
                request_timeout=60,
            )
        finally:
            await self.worker.stop()
            await self.bot.close_session()

    async def start_command(self, message: types.Message) -> None:
        payload = extract_command_query(message.text).lower()
        text = HELP_TEXT if payload == "help" else WELCOME_TEXT
        await self.bot.send_message(message.chat.id, text)

    async def help_command(self, message: types.Message) -> None:
        await self.bot.send_message(message.chat.id, HELP_TEXT)

    async def chat_membership(self, update: types.ChatMemberUpdated) -> None:
        if update.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
            return
        status = str(update.new_chat_member.status)
        active = status == "administrator"
        with session_scope() as session:
            chat = session.get(Chat, update.chat.id)
            if chat is None:
                chat = Chat(
                    chat_id=update.chat.id,
                    type=str(update.chat.type),
                )
                session.add(chat)
            chat.type = str(update.chat.type)
            chat.title = update.chat.title
            chat.bot_status = status
            chat.active = active
            chat.updated_at = utc_now()

        if status == "member":
            await self.bot.send_message(
                update.chat.id,
                "Make the bot an administrator to enable indexing and secure "
                "membership checks. No extra permissions are needed.",
            )
        elif active:
            await self.bot.send_message(
                update.chat.id,
                "Ready. I will index new images, GIFs, and videos. "
                "Search privately or use /search in this group.",
            )

    async def index_media(self, message: types.Message) -> None:
        if is_bot_authored_media(
            message,
            self.bot_user_id,
            index_bot_media=self.settings.index_bot_media,
        ):
            logger.debug(
                "Skipping bot-authored media: chat=%s message=%s",
                message.chat.id,
                message.message_id,
            )
            return
        if bool(getattr(message, "has_protected_content", False)):
            logger.info(
                "Skipping protected media: chat=%s message=%s",
                message.chat.id,
                message.message_id,
            )
            return
        if not await self._ensure_chat_is_indexable(message.chat):
            return

        candidate = select_media_candidate(message, self.settings.max_file_size)
        if candidate is None:
            return

        with session_scope() as session:
            existing = session.exec(
                select(StoredMessage).where(
                    StoredMessage.chat_id == message.chat.id,
                    StoredMessage.message_id == message.message_id,
                )
            ).scalars().first()
            if existing is not None:
                return

            stored_message = StoredMessage(
                chat_id=message.chat.id,
                sender_id=(
                    message.from_user.id
                    if message.from_user is not None
                    else getattr(message.sender_chat, "id", None)
                ),
                message_id=message.message_id,
                caption=message.caption,
                media_group_id=message.media_group_id,
            )
            media = Media(
                message_uuid=stored_message.message_uuid,
                file_id=candidate.file_id,
                file_unique_id=candidate.file_unique_id,
                media_type=candidate.media_type,
                mime_type=candidate.mime_type,
                file_size=candidate.file_size,
                duration=candidate.duration,
            )
            session.add(stored_message)
            session.flush()
            session.add(media)
            media_uuid = media.media_uuid

        await self.worker.enqueue(media_uuid)

    async def search(self, message: types.Message) -> None:
        if not message.text:
            return
        await self._private_search(message, message.text)

    async def private_search_command(self, message: types.Message) -> None:
        query = extract_command_query(message.text)
        if not query:
            await self.bot.send_message(
                message.chat.id,
                "Usage: /find <description>",
            )
            return
        await self._private_search(message, query)

    async def _private_search(
        self,
        message: types.Message,
        query: str,
    ) -> None:
        if message.from_user is None:
            return
        with session_scope() as session:
            known_chats = active_chat_ids(session)
        if not known_chats:
            await self.bot.send_message(
                message.chat.id,
                "No archives are available yet. Add the bot to a group as an "
                "administrator.",
            )
            return

        allowed_chats = await get_allowed_chat_ids(
            self.bot,
            message.from_user.id,
            known_chats,
            self.settings.membership_concurrency,
            self.bot_user_id,
        )
        if not allowed_chats:
            await self.bot.send_message(
                message.chat.id,
                "I could not find a shared group available for search.",
            )
            return

        with session_scope() as session:
            results = await self.search_service.search(
                session,
                query,
                allowed_chats,
            )

        sent = await self._send_search_results(message.chat.id, results)
        if sent == 0:
            await self.bot.send_message(
                message.chat.id,
                "I could not find a sufficiently similar result.",
            )

    async def group_search(self, message: types.Message) -> None:
        query = extract_command_query(message.text)
        if not query:
            await self.bot.send_message(
                message.chat.id,
                "Usage: /search <description>",
            )
            return
        if not await self._ensure_chat_is_indexable(message.chat):
            await self.bot.send_message(
                message.chat.id,
                "Group search requires the bot to be an administrator.",
            )
            return

        with session_scope() as session:
            results = await self.search_service.search(
                session,
                query,
                [message.chat.id],
            )

        sent = await self._send_search_results(message.chat.id, results)
        if sent == 0:
            await self.bot.send_message(
                message.chat.id,
                "I could not find a sufficiently similar result in this group.",
            )

    async def inline_search(self, inline_query: types.InlineQuery) -> None:
        query = " ".join((inline_query.query or "").split())
        if len(query) < INLINE_MIN_QUERY_LENGTH:
            await self.bot.answer_inline_query(
                inline_query.id,
                [],
                cache_time=0,
                is_personal=True,
                switch_pm_text="Open private search help",
                switch_pm_parameter="help",
            )
            return

        with session_scope() as session:
            known_chats = active_chat_ids(session)
        if not known_chats:
            await self.bot.answer_inline_query(
                inline_query.id,
                [],
                cache_time=0,
                is_personal=True,
                switch_pm_text="Add me to a group first",
                switch_pm_parameter="start",
            )
            return

        allowed_chats = await get_allowed_chat_ids(
            self.bot,
            inline_query.from_user.id,
            known_chats,
            self.settings.membership_concurrency,
            self.bot_user_id,
        )
        if not allowed_chats:
            await self.bot.answer_inline_query(
                inline_query.id,
                [],
                cache_time=0,
                is_personal=True,
                switch_pm_text="No shared groups available",
                switch_pm_parameter="start",
            )
            return

        with session_scope() as session:
            results = await self.search_service.search(
                session,
                query,
                allowed_chats,
            )

        inline_results = build_inline_query_results(
            results[: self.settings.search_results]
        )
        try:
            await self.bot.answer_inline_query(
                inline_query.id,
                inline_results,
                cache_time=0,
                is_personal=True,
            )
        except ApiTelegramException as error:
            logger.warning("Inline answer failed: %s", error)
            # Drop results Telegram rejects (stale file_id) and retry once.
            await self.bot.answer_inline_query(
                inline_query.id,
                [],
                cache_time=0,
                is_personal=True,
            )

    async def _send_search_results(
        self,
        destination_chat_id: int,
        results: Sequence[SearchResult],
    ) -> int:
        sent = 0
        for result in results:
            if sent >= self.settings.search_results:
                break
            try:
                await self.bot.forward_message(
                    destination_chat_id,
                    result.chat_id,
                    result.message_id,
                )
                sent += 1
            except ApiTelegramException as error:
                logger.info(
                    "Could not forward media=%s: %s",
                    result.media_uuid,
                    error,
                )
                if _is_definitively_unavailable(error):
                    self._mark_unavailable(result.media_uuid, str(error))
        return sent

    async def _ensure_chat_is_indexable(self, telegram_chat: types.Chat) -> bool:
        with session_scope() as session:
            chat = session.get(Chat, telegram_chat.id)
            if chat is not None and chat.active and chat.bot_status == "administrator":
                return True

        if self.bot_user_id is None:
            return False
        try:
            bot_member = await self.bot.get_chat_member(
                telegram_chat.id,
                self.bot_user_id,
            )
            status = str(bot_member.status)
        except Exception:
            logger.warning(
                "Could not verify bot membership in chat=%s",
                telegram_chat.id,
                exc_info=True,
            )
            return False

        active = status == "administrator"
        with session_scope() as session:
            chat = session.get(Chat, telegram_chat.id)
            if chat is None:
                chat = Chat(
                    chat_id=telegram_chat.id,
                    type=str(telegram_chat.type),
                )
                session.add(chat)
            chat.title = telegram_chat.title
            chat.type = str(telegram_chat.type)
            chat.bot_status = status
            chat.active = active
            chat.updated_at = utc_now()
        return active

    async def _reconcile_known_chats(self) -> None:
        if self.bot_user_id is None:
            return
        with session_scope() as session:
            chat_ids = list(session.exec(select(Chat.chat_id)).scalars())
        semaphore = asyncio.Semaphore(self.settings.membership_concurrency)

        async def inspect_chat(chat_id: int) -> tuple[int, str | None]:
            async with semaphore:
                try:
                    member = await self.bot.get_chat_member(chat_id, self.bot_user_id)
                    return chat_id, str(member.status)
                except Exception:
                    logger.warning(
                        "Could not reconcile bot membership in chat=%s",
                        chat_id,
                        exc_info=True,
                    )
                    return chat_id, None

        statuses = await asyncio.gather(*(inspect_chat(chat_id) for chat_id in chat_ids))
        with session_scope() as session:
            for chat_id, status in statuses:
                chat = session.get(Chat, chat_id)
                if chat is None:
                    continue
                chat.bot_status = status
                chat.active = status == "administrator"
                chat.updated_at = utc_now()

    def _mark_unavailable(self, media_uuid, error: str) -> None:
        with session_scope() as session:
            media = session.get(Media, media_uuid)
            if media is None:
                return
            media.status = ProcessingStatus.UNAVAILABLE
            media.error = error[:2_000]
            media.updated_at = utc_now()


def extract_command_query(text: str | None) -> str:
    if not text:
        return ""
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else ""


def build_inline_query_results(
    results: Sequence[SearchResult],
) -> list[types.InlineQueryResultCachedBase]:
    inline_results: list[types.InlineQueryResultCachedBase] = []
    for result in results:
        result_id = result.media_uuid.hex
        title = result.title or result.media_type.title()
        description = result.chat_title or None
        media_type = result.media_type

        if media_type == MediaType.IMAGE:
            inline_results.append(
                types.InlineQueryResultCachedPhoto(
                    id=result_id,
                    photo_file_id=result.file_id,
                    title=title,
                    description=description,
                )
            )
        elif media_type == MediaType.ANIMATION:
            inline_results.append(
                types.InlineQueryResultCachedMpeg4Gif(
                    id=result_id,
                    mpeg4_file_id=result.file_id,
                    title=title,
                    description=description,
                )
            )
        elif media_type == MediaType.VIDEO:
            inline_results.append(
                types.InlineQueryResultCachedVideo(
                    id=result_id,
                    video_file_id=result.file_id,
                    title=title,
                    description=description,
                )
            )
        else:
            logger.debug(
                "Skipping unsupported inline media_type=%s media=%s",
                media_type,
                result.media_uuid,
            )
    return inline_results


def is_bot_authored_media(
    message: types.Message,
    bot_user_id: int | None,
    *,
    index_bot_media: bool = False,
) -> bool:
    """Return True when media should be skipped as bot-authored.

    This bot's own messages and messages sent through this bot are always
    skipped to avoid feedback loops. Other bots are skipped unless
    ``index_bot_media`` is enabled.
    """
    author = getattr(message, "from_user", None)
    if author is not None:
        author_id = getattr(author, "id", None)
        if bot_user_id is not None and author_id == bot_user_id:
            return True
        if bool(getattr(author, "is_bot", False)) and not index_bot_media:
            return True

    via_bot = getattr(message, "via_bot", None)
    return bool(
        bot_user_id is not None
        and via_bot is not None
        and getattr(via_bot, "id", None) == bot_user_id
    )


def select_media_candidate(
    message: types.Message,
    max_file_size: int,
) -> MediaCandidate | None:
    if message.photo:
        eligible = [
            photo
            for photo in message.photo
            if photo.file_size is None or photo.file_size <= max_file_size
        ]
        if not eligible:
            return None
        photo = max(
            eligible,
            key=lambda item: (
                item.file_size or 0,
                item.width * item.height,
            ),
        )
        return MediaCandidate(
            file_id=photo.file_id,
            file_unique_id=photo.file_unique_id,
            media_type=MediaType.IMAGE,
            mime_type="image/jpeg",
            file_size=photo.file_size,
            duration=None,
        )

    if message.video is not None:
        video = message.video
        if video.file_size is not None and video.file_size > max_file_size:
            return None
        return MediaCandidate(
            file_id=video.file_id,
            file_unique_id=video.file_unique_id,
            media_type=MediaType.VIDEO,
            mime_type=getattr(video, "mime_type", None) or "video/mp4",
            file_size=video.file_size,
            duration=getattr(video, "duration", None),
        )

    if message.animation is not None:
        animation = message.animation
        if (
            animation.file_size is not None
            and animation.file_size > max_file_size
        ):
            return None
        return MediaCandidate(
            file_id=animation.file_id,
            file_unique_id=animation.file_unique_id,
            media_type=MediaType.ANIMATION,
            mime_type=getattr(animation, "mime_type", None) or "video/mp4",
            file_size=animation.file_size,
            duration=getattr(animation, "duration", None),
        )

    document = message.document
    if document is None:
        return None
    if document.file_size is not None and document.file_size > max_file_size:
        return None
    mime_type = (document.mime_type or "").lower()
    if mime_type.startswith("image/"):
        media_type = MediaType.IMAGE
    elif mime_type.startswith("video/"):
        media_type = MediaType.VIDEO
    else:
        return None
    return MediaCandidate(
        file_id=document.file_id,
        file_unique_id=document.file_unique_id,
        media_type=media_type,
        mime_type=mime_type,
        file_size=document.file_size,
        duration=None,
    )


def _is_definitively_unavailable(error: ApiTelegramException) -> bool:
    description = str(error).lower()
    return error.error_code in {400, 403, 406} and any(
        marker in description
        for marker in (
            "message to forward not found",
            "message_id_invalid",
            "chat_forwards_restricted",
            "protected content",
        )
    )

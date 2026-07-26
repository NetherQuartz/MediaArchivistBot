import asyncio
import logging
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import select
from telebot import types
from telebot.async_telebot import AsyncTeleBot

from .config import Settings, get_settings
from .database import session_scope
from .llm_api import (
    PROMPT_VERSION,
    EmbeddingService,
    VisionService,
    build_search_text,
)
from .media_processing import MediaProcessor
from .models import Media, Message, ProcessingStatus, utc_now

logger = logging.getLogger(__name__)


class IndexWorker:
    def __init__(
        self,
        bot: AsyncTeleBot,
        settings: Settings | None = None,
    ) -> None:
        self.bot = bot
        self.settings = settings or get_settings()
        self.vision = VisionService(self.settings)
        self.embeddings = EmbeddingService(self.settings)
        self.processor = MediaProcessor(self.settings)
        self.queue: asyncio.Queue[uuid.UUID | None] = asyncio.Queue()
        self.tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        pending = self._recover_pending()
        self.tasks = [
            asyncio.create_task(self._run(), name=f"index-worker-{index}")
            for index in range(self.settings.indexing_workers)
        ]
        for media_uuid in pending:
            await self.queue.put(media_uuid)
        logger.info(
            "Index workers started: workers=%s recovered=%s",
            len(self.tasks),
            len(pending),
        )

    async def stop(self) -> None:
        for _ in self.tasks:
            await self.queue.put(None)
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def enqueue(self, media_uuid: uuid.UUID) -> None:
        await self.queue.put(media_uuid)

    def _recover_pending(self) -> list[uuid.UUID]:
        with session_scope() as session:
            media_rows = session.exec(
                select(Media).where(
                    Media.status.in_(
                        [
                            ProcessingStatus.PENDING,
                            ProcessingStatus.PROCESSING,
                        ]
                    )
                )
            ).scalars().all()
            for media in media_rows:
                media.status = ProcessingStatus.PENDING
                media.error = None
                media.updated_at = utc_now()
            return [media.media_uuid for media in media_rows]

    async def _run(self) -> None:
        while True:
            media_uuid = await self.queue.get()
            try:
                if media_uuid is None:
                    return
                await self._process(media_uuid)
            except Exception:
                logger.exception("Unhandled index worker error for media=%s", media_uuid)
                if media_uuid is not None:
                    self._save_failure(media_uuid, "Unhandled worker error")
            finally:
                self.queue.task_done()

    async def _process(self, media_uuid: uuid.UUID) -> None:
        with session_scope() as session:
            media = session.get(Media, media_uuid)
            if media is None or media.status not in {
                ProcessingStatus.PENDING,
                ProcessingStatus.PROCESSING,
            }:
                return
            message = session.get(Message, media.message_uuid)
            if message is None:
                media.status = ProcessingStatus.FAILED
                media.error = "Source message metadata is missing"
                return

            duplicate = None
            if media.file_unique_id:
                duplicate = session.exec(
                    select(Media)
                    .where(
                        Media.file_unique_id == media.file_unique_id,
                        Media.media_uuid != media.media_uuid,
                        Media.status == ProcessingStatus.READY,
                        Media.embedding_model == self.settings.embedding_model,
                    )
                    .limit(1)
                ).scalars().first()
            if duplicate is not None:
                self._copy_index(duplicate, media)
                source = (message.chat_id, message.message_id)
                reused = True
            else:
                media.status = ProcessingStatus.PROCESSING
                media.error = None
                media.updated_at = utc_now()
                source = (message.chat_id, message.message_id)
                reused = False
                file_id = media.file_id
                media_type = media.media_type
                caption = message.caption
                mime_type = media.mime_type

        if reused:
            await self._react(*source)
            return

        temp_path: Path | None = None
        try:
            file_info = await self.bot.get_file(file_id)
            file_data = await self.bot.download_file(file_info.file_path)
            if len(file_data) > self.settings.max_file_size:
                raise ValueError("Downloaded file exceeds configured size limit")

            suffix = _suffix_for(mime_type, media_type)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=suffix,
                dir=self.settings.temp_dir,
                delete=False,
            ) as temp_file:
                temp_file.write(file_data)
                temp_path = Path(temp_file.name)

            prepared = await self.processor.prepare(temp_path, media_type)
            description = await self.vision.describe(
                prepared.images,
                media_type=media_type,
                caption=caption,
                transcript=prepared.transcript,
            )
            search_text = build_search_text(
                description,
                caption=caption,
                transcript=prepared.transcript,
            )
            embedding = (await self.embeddings.embed_documents([search_text]))[0]

            with session_scope() as session:
                media = session.get(Media, media_uuid)
                if media is None:
                    return
                media.description = description.model_dump(mode="json")
                media.transcript = prepared.transcript
                media.search_text = search_text
                media.embedding = embedding
                media.duration = prepared.duration or media.duration
                media.status = ProcessingStatus.READY
                media.error = None
                media.model_version = self.settings.vision_model
                media.embedding_model = self.settings.embedding_model
                media.prompt_version = PROMPT_VERSION
                media.updated_at = utc_now()
            await self._react(*source)
            logger.info("Media indexed: media=%s chat=%s", media_uuid, source[0])
        except Exception as error:
            logger.exception("Media indexing failed: media=%s", media_uuid)
            self._save_failure(media_uuid, str(error))
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _copy_index(self, source: Media, target: Media) -> None:
        target.description = source.description
        target.transcript = source.transcript
        target.search_text = source.search_text
        target.embedding = source.embedding
        target.duration = source.duration
        target.status = ProcessingStatus.READY
        target.error = None
        target.model_version = source.model_version
        target.embedding_model = source.embedding_model
        target.prompt_version = source.prompt_version
        target.updated_at = utc_now()

    def _save_failure(self, media_uuid: uuid.UUID, error: str) -> None:
        with session_scope() as session:
            media = session.get(Media, media_uuid)
            if media is None:
                return
            media.status = ProcessingStatus.FAILED
            media.error = error[:2_000]
            media.updated_at = utc_now()

    async def _react(self, chat_id: int, message_id: int) -> None:
        try:
            await self.bot.set_message_reaction(
                chat_id,
                message_id,
                [types.ReactionTypeEmoji("✍️")],
            )
        except Exception:
            logger.debug(
                "Could not set indexing reaction: chat=%s message=%s",
                chat_id,
                message_id,
                exc_info=True,
            )


def _suffix_for(mime_type: str | None, media_type: str) -> str:
    if mime_type:
        subtype = mime_type.split("/", maxsplit=1)[-1].split(";", maxsplit=1)[0]
        if subtype in {"jpeg", "jpg", "png", "webp", "gif", "mp4", "webm", "quicktime"}:
            return ".jpg" if subtype == "jpeg" else f".{subtype}"
    return ".jpg" if media_type == "image" else ".mp4"

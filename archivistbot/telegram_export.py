import asyncio
import hashlib
import json
import logging
import mimetypes
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import select

from .config import Settings, get_settings
from .database import session_scope
from .llm_api import (
    PROMPT_VERSION,
    EmbeddingService,
    VisionService,
    build_search_text,
)
from .media_processing import MediaProcessor
from .models import (
    EXPORT_FILE_ID_PREFIX,
    Chat,
    Media,
    MediaType,
    Message,
    ProcessingStatus,
    utc_now,
)

logger = logging.getLogger(__name__)

_MISSING_FILE_PREFIX = "(File not included"


@dataclass(frozen=True)
class ExportMedia:
    message_id: int
    sender_id: int | None
    sent_at: datetime
    caption: str | None
    path: Path
    media_type: str
    mime_type: str
    file_size: int
    duration: int | None


@dataclass
class ImportResult:
    candidates: int = 0
    existing: int = 0
    imported: int = 0
    reused: int = 0
    failed: int = 0


def load_export_media(
    export_root: Path,
    *,
    chat_id: int | None = None,
    after_message_id: int = 0,
    media_types: set[str] | None = None,
) -> tuple[int, list[ExportMedia]]:
    export_path = export_root / "result.json"
    with export_path.open(encoding="utf-8") as export_file:
        export_data = json.load(export_file)

    resolved_chat_id = chat_id or infer_export_chat_id(export_data)
    media = []
    for message in export_data.get("messages", []):
        item = parse_export_message(
            message,
            export_root=export_root,
            after_message_id=after_message_id,
        )
        if item is not None and (
            media_types is None or item.media_type in media_types
        ):
            media.append(item)
    return resolved_chat_id, media


def infer_export_chat_id(export_data: dict[str, Any]) -> int:
    export_id = export_data.get("id")
    export_type = str(export_data.get("type") or "")
    if (
        not isinstance(export_id, int)
        or export_id <= 0
        or ("supergroup" not in export_type and "channel" not in export_type)
    ):
        raise ValueError(
            "Could not infer a Bot API chat id; pass --chat-id explicitly"
        )
    return int(f"-100{export_id}")


def parse_export_message(
    message: dict[str, Any],
    *,
    export_root: Path,
    after_message_id: int = 0,
) -> ExportMedia | None:
    message_id = message.get("id")
    if (
        message.get("type") != "message"
        or not isinstance(message_id, int)
        or message_id <= after_message_id
        or message_id <= 0
    ):
        return None

    relative_path, media_type, mime_type = _media_source(message)
    if relative_path is None or relative_path.startswith(_MISSING_FILE_PREFIX):
        return None

    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        logger.warning("Skipping unsafe export path for message=%s", message_id)
        return None
    media_path = export_root.joinpath(*pure_path.parts)
    if not media_path.is_file():
        logger.warning("Skipping missing export file for message=%s", message_id)
        return None

    try:
        timestamp = int(message["date_unixtime"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Skipping export message without a valid date: %s", message_id)
        return None

    duration = message.get("duration_seconds")
    return ExportMedia(
        message_id=message_id,
        sender_id=_sender_id(message.get("from_id")),
        sent_at=datetime.fromtimestamp(timestamp, UTC),
        caption=_flatten_text(message.get("text")).strip() or None,
        path=media_path,
        media_type=media_type,
        mime_type=mime_type,
        file_size=media_path.stat().st_size,
        duration=duration if isinstance(duration, int) else None,
    )


def _media_source(message: dict[str, Any]) -> tuple[str | None, str, str]:
    photo = message.get("photo")
    if isinstance(photo, str):
        return photo, MediaType.IMAGE, "image/jpeg"

    file_path = message.get("file")
    if not isinstance(file_path, str) or message.get("media_type") == "sticker":
        return None, "", ""

    exported_type = str(message.get("media_type") or "")
    mime_type = str(message.get("mime_type") or "").lower()
    if not mime_type:
        mime_type = (mimetypes.guess_type(file_path)[0] or "").lower()

    if exported_type == "animation":
        return file_path, MediaType.ANIMATION, mime_type or "video/mp4"
    if mime_type.startswith("image/"):
        return file_path, MediaType.IMAGE, mime_type
    if mime_type.startswith("video/"):
        return file_path, MediaType.VIDEO, mime_type
    return None, "", ""


def _sender_id(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    if value.startswith("user") and value[4:].isdigit():
        return int(value[4:])
    if value.startswith("channel") and value[7:].isdigit():
        return int(f"-100{value[7:]}")
    return None


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_flatten_text(item) for item in value)
    if isinstance(value, dict):
        return _flatten_text(value.get("text"))
    return ""


async def import_telegram_export(
    export_root: Path,
    *,
    chat_id: int | None = None,
    after_message_id: int = 0,
    media_types: set[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    settings: Settings | None = None,
) -> ImportResult:
    runtime_settings = settings or get_settings()
    resolved_chat_id, candidates = await asyncio.to_thread(
        load_export_media,
        export_root,
        chat_id=chat_id,
        after_message_id=after_message_id,
        media_types=media_types,
    )
    result = ImportResult(candidates=len(candidates))

    with session_scope() as session:
        chat = session.get(Chat, resolved_chat_id)
        if chat is None or not chat.active or chat.bot_status != "administrator":
            raise ValueError(
                "The target chat must already be active with this bot as administrator"
            )
        existing_ids = set(
            session.exec(
                select(Message.message_id).where(Message.chat_id == resolved_chat_id)
            ).scalars()
        )

    result.existing = sum(
        candidate.message_id in existing_ids for candidate in candidates
    )
    pending = [
        candidate
        for candidate in candidates
        if candidate.message_id not in existing_ids
    ]
    if limit is not None:
        pending = pending[:limit]
    if dry_run or not pending:
        return result

    runtime_settings.validate_runtime()
    runtime_settings.temp_dir.mkdir(parents=True, exist_ok=True)
    processor = MediaProcessor(runtime_settings)
    vision = VisionService(runtime_settings)
    embeddings = EmbeddingService(runtime_settings)
    try:
        for index, candidate in enumerate(pending, start=1):
            try:
                reused = await _import_candidate(
                    candidate,
                    chat_id=resolved_chat_id,
                    processor=processor,
                    vision=vision,
                    embeddings=embeddings,
                    settings=runtime_settings,
                )
                result.imported += 1
                result.reused += int(reused)
                logger.info(
                    "Imported Telegram export media: message=%s progress=%s/%s reused=%s",
                    candidate.message_id,
                    index,
                    len(pending),
                    reused,
                )
            except Exception:
                result.failed += 1
                logger.exception(
                    "Telegram export import failed: message=%s",
                    candidate.message_id,
                )
    finally:
        await vision.client.close()
    return result


async def _import_candidate(
    candidate: ExportMedia,
    *,
    chat_id: int,
    processor: MediaProcessor,
    vision: VisionService,
    embeddings: EmbeddingService,
    settings: Settings,
) -> bool:
    digest = await asyncio.to_thread(_sha256, candidate.path)
    file_unique_id = f"{EXPORT_FILE_ID_PREFIX}sha256:{digest}"

    with session_scope() as session:
        duplicate = session.exec(
            select(Media)
            .where(
                Media.file_unique_id == file_unique_id,
                Media.status == ProcessingStatus.READY,
                Media.embedding_model == settings.embedding_model,
            )
            .limit(1)
        ).scalars().first()
        duplicate_index = (
            {
                "description": duplicate.description,
                "transcript": duplicate.transcript,
                "search_text": duplicate.search_text,
                "embedding": duplicate.embedding,
                "duration": duplicate.duration,
                "model_version": duplicate.model_version,
                "prompt_version": duplicate.prompt_version,
            }
            if duplicate is not None
            else None
        )

    if duplicate_index is not None:
        description = duplicate_index["description"]
        transcript = duplicate_index["transcript"]
        search_text = duplicate_index["search_text"]
        embedding = duplicate_index["embedding"]
        duration = duplicate_index["duration"] or candidate.duration
        model_version = duplicate_index["model_version"]
        prompt_version = duplicate_index["prompt_version"]
        reused = True
    else:
        temporary_path = await asyncio.to_thread(
            _copy_to_temporary_path,
            candidate.path,
            settings.temp_dir,
        )
        try:
            prepared = await processor.prepare(temporary_path, candidate.media_type)
            media_description = await vision.describe(
                prepared.images,
                media_type=candidate.media_type,
                caption=candidate.caption,
                transcript=prepared.transcript,
            )
            description = media_description.model_dump(mode="json")
            transcript = prepared.transcript
            search_text = build_search_text(
                media_description,
                caption=candidate.caption,
                transcript=prepared.transcript,
            )
            embedding = (await embeddings.embed_documents([search_text]))[0]
            duration = prepared.duration or candidate.duration
            model_version = settings.vision_model
            prompt_version = PROMPT_VERSION
            reused = False
        finally:
            temporary_path.unlink(missing_ok=True)

    stored_message = Message(
        chat_id=chat_id,
        sender_id=candidate.sender_id,
        message_id=candidate.message_id,
        caption=candidate.caption,
        media_group_id=None,
        add_date=candidate.sent_at,
    )
    now = utc_now()
    media = Media(
        message_uuid=stored_message.message_uuid,
        file_id=file_unique_id,
        file_unique_id=file_unique_id,
        media_type=candidate.media_type,
        mime_type=candidate.mime_type,
        file_size=candidate.file_size,
        duration=duration,
        description=description,
        transcript=transcript,
        search_text=search_text,
        embedding=embedding,
        status=ProcessingStatus.READY,
        model_version=model_version,
        embedding_model=settings.embedding_model,
        prompt_version=prompt_version,
        add_date=candidate.sent_at,
        updated_at=now,
    )
    with session_scope() as session:
        session.add(stored_message)
        session.flush()
        session.add(media)
    return reused


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as media_file:
        for chunk in iter(lambda: media_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_to_temporary_path(source: Path, temp_dir: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        suffix=source.suffix,
        dir=temp_dir,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        shutil.copyfile(source, temporary_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path

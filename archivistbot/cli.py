import argparse
import asyncio
import logging
from pathlib import Path

from sqlalchemy import or_, select

from .config import get_settings
from .database import session_scope
from .llm_api import EmbeddingService
from .models import Media, ProcessingStatus, utc_now
from .telegram_export import import_telegram_export


def requeue(*, all_media: bool) -> int:
    with session_scope() as session:
        query = select(Media)
        if not all_media:
            query = query.where(Media.status == ProcessingStatus.FAILED)
        media_rows = session.exec(query).scalars().all()
        for media in media_rows:
            media.status = ProcessingStatus.PENDING
            media.error = None
            if all_media:
                media.description = None
                media.transcript = None
                media.search_text = None
                media.embedding = None
                media.model_version = None
                media.embedding_model = None
                media.prompt_version = None
            media.updated_at = utc_now()
        return len(media_rows)


async def reembed(batch_size: int = 32) -> int:
    service = EmbeddingService()
    processed = 0
    while True:
        with session_scope() as session:
            media_rows = session.exec(
                select(Media)
                .where(
                    Media.status == ProcessingStatus.READY,
                    Media.search_text.is_not(None),
                    or_(
                        Media.embedding.is_(None),
                        Media.embedding_model.is_(None),
                        Media.embedding_model != service.settings.embedding_model,
                    ),
                )
                .limit(batch_size)
            ).scalars().all()
            if not media_rows:
                return processed
            media_ids = [media.media_uuid for media in media_rows]
            texts = [media.search_text for media in media_rows]

        embeddings = await service.embed_documents(texts)
        with session_scope() as session:
            for media_uuid, embedding in zip(media_ids, embeddings, strict=True):
                media = session.get(Media, media_uuid)
                if media is not None:
                    media.embedding = embedding
                    media.embedding_model = service.settings.embedding_model
                    media.updated_at = utc_now()
                    processed += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaArchivist maintenance commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    requeue_parser = subparsers.add_parser(
        "requeue",
        help="Queue failed media for processing on the next bot start",
    )
    requeue_parser.add_argument(
        "--all",
        action="store_true",
        help="Discard all indexes and describe every media item again",
    )

    reembed_parser = subparsers.add_parser(
        "reembed",
        help="Create missing embeddings from existing search text",
    )
    reembed_parser.add_argument("--batch-size", type=int, default=32)

    import_parser = subparsers.add_parser(
        "import-telegram-export",
        help="Index local media from a Telegram Desktop JSON export",
    )
    import_parser.add_argument("path", type=Path)
    import_parser.add_argument(
        "--chat-id",
        type=int,
        help="Bot API chat id; inferred for supergroup and channel exports",
    )
    import_parser.add_argument(
        "--after-message-id",
        type=int,
        default=0,
        help="Ignore messages at or below this id",
    )
    import_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of new media items to import",
    )
    import_parser.add_argument(
        "--media-type",
        action="append",
        choices=["image", "video", "animation"],
        help="Only import this media type; may be repeated",
    )
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the export and report counts without indexing",
    )

    arguments = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, get_settings().logging_level.upper(), logging.INFO)
    )
    if arguments.command == "requeue":
        count = requeue(all_media=arguments.all)
        print(f"Updated media rows: {count}")
    elif arguments.command == "reembed":
        count = asyncio.run(reembed(arguments.batch_size))
        print(f"Updated media rows: {count}")
    else:
        result = asyncio.run(
            import_telegram_export(
                arguments.path,
                chat_id=arguments.chat_id,
                after_message_id=arguments.after_message_id,
                media_types=(
                    set(arguments.media_type) if arguments.media_type else None
                ),
                limit=arguments.limit,
                dry_run=arguments.dry_run,
            )
        )
        print(
            "Import result: "
            f"candidates={result.candidates} "
            f"existing={result.existing} "
            f"imported={result.imported} "
            f"reused={result.reused} "
            f"failed={result.failed}"
        )


if __name__ == "__main__":
    main()

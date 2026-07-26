import argparse
import asyncio

from sqlalchemy import or_, select

from .database import session_scope
from .llm_api import EmbeddingService
from .models import Media, ProcessingStatus, utc_now


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

    arguments = parser.parse_args()
    if arguments.command == "requeue":
        count = requeue(all_media=arguments.all)
    else:
        count = asyncio.run(reembed(arguments.batch_size))
    print(f"Updated media rows: {count}")


if __name__ == "__main__":
    main()

import uuid
from types import SimpleNamespace

from telebot import types

from archivistbot.bot import (
    build_inline_query_results,
    extract_command_query,
    is_bot_authored_media,
    select_media_candidate,
)
from archivistbot.models import Media, MediaType
from archivistbot.search import SearchResult, media_result_title


def message(**overrides):
    values = {
        "photo": None,
        "video": None,
        "animation": None,
        "document": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_selects_largest_photo_within_limit() -> None:
    small = SimpleNamespace(
        file_id="small",
        file_unique_id="same",
        file_size=1_000,
        width=320,
        height=240,
    )
    large = SimpleNamespace(
        file_id="large",
        file_unique_id="same",
        file_size=5_000,
        width=1920,
        height=1080,
    )

    candidate = select_media_candidate(
        message(photo=[small, large]),
        max_file_size=10_000,
    )

    assert candidate is not None
    assert candidate.file_id == "large"
    assert candidate.media_type == MediaType.IMAGE


def test_rejects_unsupported_document() -> None:
    document = SimpleNamespace(
        file_id="archive",
        file_unique_id="archive-unique",
        file_size=1_000,
        mime_type=None,
    )

    assert (
        select_media_candidate(
            message(document=document),
            max_file_size=10_000,
        )
        is None
    )


def test_accepts_video_document() -> None:
    document = SimpleNamespace(
        file_id="video",
        file_unique_id="video-unique",
        file_size=1_000,
        mime_type="video/webm",
    )

    candidate = select_media_candidate(
        message(document=document),
        max_file_size=10_000,
    )

    assert candidate is not None
    assert candidate.media_type == MediaType.VIDEO
    assert candidate.mime_type == "video/webm"


def test_animation_is_not_treated_as_video_with_audio() -> None:
    animation = SimpleNamespace(
        file_id="gif",
        file_unique_id="gif-unique",
        file_size=1_000,
        mime_type="video/mp4",
        duration=3,
    )

    candidate = select_media_candidate(
        message(animation=animation),
        max_file_size=10_000,
    )

    assert candidate is not None
    assert candidate.media_type == MediaType.ANIMATION


def test_extracts_group_search_query() -> None:
    assert extract_command_query("/search cat in deep snow") == "cat in deep snow"
    assert extract_command_query("/find@media_bot black cat") == "black cat"
    assert extract_command_query("/search") == ""


def test_bot_authored_media_is_excluded() -> None:
    this_bot_message = SimpleNamespace(
        from_user=SimpleNamespace(id=99, is_bot=True),
        via_bot=None,
    )
    other_bot_message = SimpleNamespace(
        from_user=SimpleNamespace(id=77, is_bot=True),
        via_bot=None,
    )
    human_message = SimpleNamespace(
        from_user=SimpleNamespace(id=42, is_bot=False),
        via_bot=None,
    )
    inline_result = SimpleNamespace(
        from_user=SimpleNamespace(id=42, is_bot=False),
        via_bot=SimpleNamespace(id=99),
    )

    assert is_bot_authored_media(this_bot_message, 99)
    assert is_bot_authored_media(other_bot_message, 99)
    assert is_bot_authored_media(inline_result, 99)
    assert not is_bot_authored_media(human_message, 99)

    assert is_bot_authored_media(this_bot_message, 99, index_bot_media=True)
    assert not is_bot_authored_media(other_bot_message, 99, index_bot_media=True)
    assert is_bot_authored_media(inline_result, 99, index_bot_media=True)
    assert not is_bot_authored_media(human_message, 99, index_bot_media=True)


def test_media_result_title_prefers_summary() -> None:
    media = Media(
        file_id="f",
        media_type=MediaType.IMAGE,
        description={"summary": "Stilgar as it was written"},
        search_text="Description: fallback",
    )
    assert media_result_title(media) == "Stilgar as it was written"


def test_builds_cached_inline_results_by_media_type() -> None:
    photo_id = uuid.uuid4()
    video_id = uuid.uuid4()
    animation_id = uuid.uuid4()
    results = [
        SearchResult(
            media_uuid=photo_id,
            chat_id=-1,
            message_id=1,
            file_id="photo-file",
            file_unique_id="photo",
            media_type=MediaType.IMAGE,
            mime_type="image/jpeg",
            title="photo title",
            chat_title="Memes",
            score=1.0,
            cosine_distance=0.1,
        ),
        SearchResult(
            media_uuid=video_id,
            chat_id=-1,
            message_id=2,
            file_id="video-file",
            file_unique_id="video",
            media_type=MediaType.VIDEO,
            mime_type="video/mp4",
            title="video title",
            chat_title="Memes",
            score=0.9,
            cosine_distance=0.2,
        ),
        SearchResult(
            media_uuid=animation_id,
            chat_id=-1,
            message_id=3,
            file_id="gif-file",
            file_unique_id="gif",
            media_type=MediaType.ANIMATION,
            mime_type="video/mp4",
            title="gif title",
            chat_title=None,
            score=0.8,
            cosine_distance=0.3,
        ),
    ]

    inline_results = build_inline_query_results(results)

    assert [type(item) for item in inline_results] == [
        types.InlineQueryResultCachedPhoto,
        types.InlineQueryResultCachedVideo,
        types.InlineQueryResultCachedMpeg4Gif,
    ]
    assert inline_results[0].id == photo_id.hex
    assert inline_results[1].title == "video title"
    assert inline_results[2].mpeg4_file_id == "gif-file"

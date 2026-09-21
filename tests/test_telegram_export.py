import json
from pathlib import Path

import pytest

from archivistbot.models import EXPORT_FILE_ID_PREFIX, MediaType
from archivistbot.telegram_export import (
    infer_export_chat_id,
    load_export_media,
)


def test_loads_supported_positive_media_from_supergroup_export(
    tmp_path: Path,
) -> None:
    photos = tmp_path / "photos"
    files = tmp_path / "files"
    photos.mkdir()
    files.mkdir()
    (photos / "photo.jpg").write_bytes(b"photo")
    (files / "large-video.bin").write_bytes(b"video")
    export = {
        "id": 1980290791,
        "type": "private_supergroup",
        "messages": [
            {
                "id": -1,
                "type": "message",
                "date_unixtime": "1",
                "photo": "photos/photo.jpg",
            },
            {
                "id": 1,
                "type": "message",
                "date_unixtime": "2",
                "file": "files/sticker.webm",
                "media_type": "sticker",
                "mime_type": "video/webm",
            },
            {
                "id": 2,
                "type": "message",
                "date_unixtime": "3",
                "from_id": "user42",
                "file": "files/large-video.bin",
                "media_type": "file",
                "mime_type": "video/mp4",
                "file_size": 30_000_000,
                "text": [
                    "A ",
                    {"type": "bold", "text": "caption"},
                ],
            },
            {
                "id": 3,
                "type": "message",
                "date_unixtime": "4",
                "photo": "photos/photo.jpg",
                "text": "",
            },
            {
                "id": 4,
                "type": "message",
                "date_unixtime": "5",
                "photo": "../outside.jpg",
            },
            {
                "id": 5,
                "type": "message",
                "date_unixtime": "6",
                "file": "(File not included. Change data exporting settings to download.)",
                "media_type": "video_file",
                "mime_type": "video/mp4",
            },
        ],
    }
    (tmp_path / "result.json").write_text(json.dumps(export), encoding="utf-8")

    chat_id, media = load_export_media(tmp_path)

    assert chat_id == -1001980290791
    assert [item.message_id for item in media] == [2, 3]
    assert media[0].media_type == MediaType.VIDEO
    assert media[0].caption == "A caption"
    assert media[0].sender_id == 42
    assert media[1].media_type == MediaType.IMAGE

    _, images = load_export_media(tmp_path, media_types={MediaType.IMAGE})

    assert [item.message_id for item in images] == [3]


def test_explicit_chat_id_supports_exports_without_supergroup_id(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.json").write_text(
        json.dumps({"id": 7, "type": "private_group", "messages": []}),
        encoding="utf-8",
    )

    chat_id, media = load_export_media(tmp_path, chat_id=-7)

    assert chat_id == -7
    assert media == []


def test_non_supergroup_chat_id_cannot_be_inferred() -> None:
    with pytest.raises(ValueError, match="pass --chat-id"):
        infer_export_chat_id({"id": 7, "type": "private_group"})


def test_export_file_id_prefix_is_not_a_real_telegram_file_id() -> None:
    assert EXPORT_FILE_ID_PREFIX == "telegram-export:"

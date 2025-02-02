import os
import asyncio
import logging
import requests
import tempfile

from telebot.async_telebot import AsyncTeleBot
from telebot import types, logger

from sqlmodel import select
from fast_depends import inject
from sqlalchemy.exc import IntegrityError

from .database import File, Chat, Message, User, MediaType, SessionType
from .llm_api import describe_photo, describe_video, get_embedding

logger.setLevel(logging.DEBUG if os.getenv("LOGGING_LEVEL") == "DEBUG" else logging.INFO)
bot = AsyncTeleBot(token=os.getenv("TG_TOKEN"))

welcome_text = """
Hello!
"""

MAX_FILE_SIZE = 20_000_000


@bot.message_handler(commands=["start"])
async def start(message: types.Message) -> None:
    await bot.send_message(message.chat.id, text=welcome_text, parse_mode="markdown")


@bot.message_handler(chat_types=["group", "supergroup", "channel"], content_types=["photo", "video", "animation", "document"])
@inject
async def index_media(message: types.Message, session: SessionType) -> None:
    logger.info(f"Got media: {message.chat.type=} {message.chat.id=} {message.id=} {message.content_type=}")

    chat = session.get(Chat, message.chat.id)
    if not chat:
        chat = session.add(Chat(chat_id=message.chat.id, type=message.chat.type))
        session.commit()
    msg = Message(chat_id=message.chat.id, sender_id=message.from_user.id, message_id=message.id)
    session.add(msg)
    session.commit()

    files: list[File] = []

    if message.photo:
        for photo in message.photo:
            logger.info(f"{photo.file_size=} {photo.file_id=}")
            file = session.get(File, photo.file_id)
            if file is None and photo.file_size < MAX_FILE_SIZE:
                file = File(file_id=photo.file_id, media_type=MediaType.image, message_uuid=msg.message_uuid)
                files.append(file)

    if ((video := message.video) or (video := message.animation)) and video.file_size < MAX_FILE_SIZE:
        logger.info(f"{video.file_size=} {video.file_id=} {video.file_name=}")
        file = session.get(File, video.file_id)
        if file is None and video.file_size < MAX_FILE_SIZE:
            file = File(file_id=video.file_id, media_type=MediaType.video, message_uuid=msg.message_uuid)
            files.append(file)

    if document := message.document:
        logger.info(f"{document.file_size=} {document.file_id=} {document.file_name=} {document.mime_type=}")
        file = session.get(File, document.file_id)
        if file is None and document.file_size < MAX_FILE_SIZE:
            if "video" in document.mime_type:
                file = File(file_id=document.file_id, media_type=MediaType.video, message_uuid=msg.message_uuid)
            if "image" in document.mime_type:
                file = File(file_id=document.file_id, media_type=MediaType.image, message_uuid=msg.message_uuid)
            files.append(file)

    for file in files:
        try:
            session.add(file)
            session.commit()
        except IntegrityError as e:
            logger.error(e)
            session.rollback()

    descriptions = []
    for file in files:
        file_url = await bot.get_file_url(file.file_id)
        file_data = requests.get(file_url).content
        match file.media_type:
            case MediaType.image:
                file.description = describe_photo(file_data)
            case MediaType.video:
                with tempfile.NamedTemporaryFile("wb", dir="/ramdisk") as temp:
                    temp.write(file_data)
                    file.description = describe_video(temp.name)
        descriptions.append(file.description)

    for embedding in await get_embedding(descriptions):
        file.embedding = embedding
    session.commit()

    if files:
        await bot.set_message_reaction(message.chat.id, message.id, [types.ReactionTypeEmoji("✍️")])
        logger.info(f"Finished media processing: added {len(files)} files")
    else:
        logger.info("Finished with no new media")


@bot.message_handler(chat_types=["private"])
@inject
async def search(message: types.Message, session: SessionType) -> None:
    query_embedding = (await get_embedding([message.text]))[0]
    results = session.exec(select(Message).join(File).order_by(File.embedding.l2_distance(query_embedding)).limit(3))
    for msg in results.unique():
        await bot.forward_message(message.chat.id, msg.chat_id, msg.message_id)


@bot.my_chat_member_handler()
@inject
async def chat_membership(upd: types.ChatMemberUpdated, session: SessionType) -> None:
    chat = session.get(Chat, upd.chat.id)
    if not chat:
        chat = session.add(Chat(chat_id=upd.chat.id, type=upd.chat.type))
    session.commit()


if __name__ == "__main__":
    asyncio.run(bot.infinity_polling())

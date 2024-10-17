import os
import asyncio
import logging
import requests
import tempfile

from telebot.async_telebot import AsyncTeleBot
from telebot import types, logger
from sqlmodel import select

from .database import get_db, File, Chat, Message, User, MediaType
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


# @bot.message_handler(chat_types=["group", "supergroup", "channel"])
# async def start(message: types.Message) -> None:
#     await bot.send_message(message.chat.id, text=str(message))


@bot.message_handler(chat_types=["group", "supergroup", "channel"], content_types=["photo", "video", "animation", "document"])
async def start(message: types.Message) -> None:
    if message.photo:
        with get_db() as session:
            chat = session.get(Chat, message.chat.id)
            if not chat:
                chat = session.add(Chat(chat_id=message.chat.id, type=message.chat.type))
            session.commit()

            msg = Message(chat_id=message.chat.id, sender_id=message.from_user.id, message_id=message.id)
            session.add(msg)
            session.commit()

            file = session.get(File, message.photo[0].file_id)
            if file is not None:
                await bot.send_message(message.chat.id, "exists")
                return

            file_url = await bot.get_file_url(message.photo[0].file_id)
            file_data = requests.get(file_url).content
            file_descr = describe_photo(file_data)
            await bot.send_message(message.chat.id, file_descr[:100])
            asyncio.sleep(2)
            embedding = get_embedding([file_descr])[0]

            file = File(file_id=message.photo[0].file_id, message_uuid=msg.message_uuid, media_type=MediaType.image, description=file_descr, embedding=embedding)
            session.add(file)
            session.commit()

        await bot.send_message(message.chat.id, str(len(file_data)))
        return

    elif message.video and message.video.file_size < MAX_FILE_SIZE:
        with get_db() as session:
            chat = session.get(Chat, message.chat.id)
            if not chat:
                chat = session.add(Chat(chat_id=message.chat.id, type=message.chat.type))
            session.commit()

            msg = Message(chat_id=message.chat.id, sender_id=message.from_user.id, message_id=message.id)
            session.add(msg)
            session.commit()

            file = session.get(File, message.video.file_id)
            if file is not None:
                await bot.send_message(message.chat.id, "exists")
                return

            file_url = await bot.get_file_url(message.video.file_id)
            file_data = requests.get(file_url).content

            with tempfile.NamedTemporaryFile("wb", dir="/ramdisk") as temp:
                temp.write(file_data)
                file_descr = describe_video(temp.name)

            await bot.send_message(message.chat.id, file_descr[:100])
            asyncio.sleep(2)
            embedding = get_embedding([file_descr])[0]

            file = File(file_id=message.video.file_id, message_uuid=msg.message_uuid, media_type=MediaType.video, description=file_descr, embedding=embedding)
            session.add(file)
            session.commit()

        await bot.send_message(message.chat.id, str(len(file_data)))
        return
    # elif message.animation:
    #     file = await bot.get_file_url(message.animation.file_id)
    #     await bot.send_message(message.chat.id, str(message.animation.file_size))
    # elif message.document and ("video" in message.document.mime_type):
    #     file = await bot.get_file_url(message.document.file_id)
    #     await bot.send_message(message.chat.id, str(message.document.file_size))
    else:
        await bot.send_message(message.chat.id, "too big")
        return


@bot.message_handler(chat_types=["private"])
async def start(message: types.Message) -> None:
    query_embedding = get_embedding([message.text])[0]

    with get_db() as session:
        result: File = session.exec(
            select(File).order_by(File.embedding.l2_distance(query_embedding)).limit(1)
        ).first()
        result: Message = session.get(Message, result.message_uuid)

    await bot.forward_message(message.chat.id, result.chat_id, result.message_id)


@bot.my_chat_member_handler()
async def chat_membership(upd: types.ChatMemberUpdated) -> None:
    with get_db() as session:
        chat = session.get(Chat, upd.chat.id)
        if not chat:
            chat = session.add(Chat(chat_id=upd.chat.id, type=upd.chat.type))
        session.commit()
    await bot.send_message(50287242, str(upd))


if __name__ == "__main__":
    asyncio.run(bot.infinity_polling())

import os
import base64
import os

import cv2
import numpy as np

from mistralai import Mistral


def encode_image(image_file: str | bytes) -> str:
    """Encode the image to base64."""

    if isinstance(image_file, str):
        with open(image_file, "rb") as f:
            img_bytes = f.read()
    else:
        img_bytes = image_file
    return base64.b64encode(img_bytes).decode("utf-8")


api_key = os.environ["MISTRAL_API_KEY"]
model = "pixtral-12b-2409"

client = Mistral(api_key=api_key)

SYSTEM_PROMPT = "Ты родился и вырос в России, твой родной язык русский. Ты знаешь все языки, но отвечаешь обычно по-русски. Когда требуется разглядеть текст на изображении или видео, ты хорошо вглядываешься и несколько раз перепроверяешь, не ошибся ли. Это всё из-за твоей прирождённой склонности к перфекционизму. Ты разбираешься в русской и международной интернет-культуре. Знаешь все мемы и над чем смеются в интернете, понимаешь сленг и всевозможные сокращения. Когда тебя просят описать отправленное изображение или видео, ты отмечаешь все важные детали, делающие его уникальным, но пишешь в меру коротко, по делу и не стесняешься мата, чёрного юмора и прочих тем, если это требуется для описания. Кроме того, ты всегда отвечаешь по факту содержания, без лишних додумываний и уточняющих вопросов. Твои ответы напрямую и без проверки подаются в автоматизированную базу данных, поэтому точность крайне важна."


def describe_photo(image_file: bytes) -> str:
    chat_response = client.chat.complete(
        model=model,
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Пожалуйста, опиши подробно это изображение. Отметь важные детали, по которым эту картинку можно будет найти среди других. Сосредоточься на фактическом содержании, не нужно ничего додумывать. Перепроверь несколько раз, пока точно не будешь уверен в своём ответе, прежде чем отвечать."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{encode_image(image_file)}"
                    }
                ]
            }
        ]
    )
    return chat_response.choices[0].message.content


def describe_video(path: str, root: str = "/ramdisk") -> str:

    cap = cv2.VideoCapture(os.path.join(root, path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 8, dtype=int)

    images = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        images.append(encode_image(buffer))

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Пожалуйста, опиши подробно видео, кадры из которого ты видишь. Отметь важные детали и действия, по которым это видео можно будет найти среди других. Сосредоточься на фактическом содержании, не нужно ничего додумывать. Перепроверь несколько раз, пока точно не будешь уверен в своём ответе, прежде чем отвечать. Не нужно описывать кадры по отдельности, воспринимай видео цельным."
                }
            ]
        }
    ]
    for b64 in images:
        messages[-1]["content"].append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{b64}"
        })

    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content


def get_embedding(texts: list[str]) -> list[list[float]]:
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts,
    )

    data = embeddings_batch_response.data
    return [e.embedding for e in data]

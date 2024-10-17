FROM python:3.12-slim

WORKDIR /usr/app

ENV TZ=Europe/Moscow

ADD requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6
RUN pip install --no-cache-dir -r requirements.txt

ADD bot bot

ENTRYPOINT [ "python", "-m", "bot" ]

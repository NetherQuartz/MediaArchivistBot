FROM python:3.12-slim

WORKDIR /usr/app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/usr/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Moscow

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini .
COPY migrations migrations
COPY archivistbot archivistbot
COPY tests tests

CMD ["python", "-m", "archivistbot"]

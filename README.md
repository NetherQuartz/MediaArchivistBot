# MediaArchivistBot

A Telegram bot for searching a meme archive collected from group chats.

The bot indexes new photos, GIFs, and videos, creates searchable descriptions,
stores embeddings and Telegram source-message IDs, and forwards matching
messages in response to private text queries or group `/search` commands. Media
files are not persisted.

## How it works

- PostgreSQL + pgvector store messages, descriptions, transcripts, and vectors.
- An OpenRouter-compatible vision model extracts OCR, characters, actions, and
  meme context as structured JSON, including English and Russian search aliases.
- Ollama runs `qwen3-embedding:0.6b` in a separate container for local,
  multilingual embeddings.
- `faster-whisper` transcribes video speech locally.
- Retrieval combines cosine similarity, PostgreSQL full-text search, and
  trigram matching.
- Before private search, the bot checks current membership through
  `getChatMember` and filters the SQL query before ranking. Group search is
  always restricted to the current group.

## Setup

1. Create a bot with BotFather and obtain its token.
2. Copy the environment template:

   ```bash
   cp .env.template .env
   ```

3. Configure at least:

   ```dotenv
   TG_TOKEN=...
   OPENROUTER_API_KEY=...
   POSTGRES_PASSWORD=choose-a-random-password
   ```

4. Start the stack:

   ```bash
   docker compose up -d
   ```

Compose pulls `ghcr.io/netherquartz/mediaarchivistbot:latest` from GitHub
Container Registry, starts PostgreSQL and Ollama, downloads the embedding
model, applies `alembic upgrade head`, and only then starts the bot. The first
start downloads approximately 639 MB of embedding-model weights. The first
processed video also downloads the local Whisper model.

For local development against a freshly built image instead of GHCR:

```bash
BOT_PULL_POLICY=build docker compose up --build -d
```

If the GHCR package is private, authenticate once before pulling:

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u USERNAME --password-stdin
```

## CI/CD

Pushes to `main` rebuild the Docker image and run unit checks. Pushing a git
tag publishes the image to GHCR with that tag and updates `latest`:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Published images:

- `ghcr.io/netherquartz/mediaarchivistbot:<git-tag>`
- `ghcr.io/netherquartz/mediaarchivistbot:latest`

On macOS, Ollama runs on the CPU inside Docker because Docker Desktop does not
expose Metal GPUs to containers. This is reproducible and sufficient for
embeddings, although a native Ollama installation is faster.

## Telegram configuration

Add the bot to a group as an **administrator**. This is required because:

1. administrators receive all new media regardless of privacy mode;
2. Telegram guarantees `getChatMember` results for other users only when the
   bot is an administrator.

The minimum available administrator privileges are sufficient. The bot only
indexes `group` and `supergroup` chats where it is an administrator. A private
message searches every group where current membership is confirmed; plain text,
`/search <query>`, and `/find <query>` are equivalent there. Inside a group,
`/search` and `/find` search only that group's archive. Inline search
(`@bot_username <query>`) also uses membership-scoped archives and shows up to
five results in Telegram's result picker without flooding the chat. Telegram
inline queries do not include the current group id, so inline mode cannot be
limited to only the group where you are typing. Both command and inline modes
return up to five relevant results.

```text
/search Stilgar as it was written
@YourArchivistBot Stilgar as it was written
```

By default, media authored by bots is excluded from indexing. Set
`INDEX_BOT_MEDIA=true` to index media posted by other bots. Messages authored
by this bot, or sent through this bot, are always skipped to prevent feedback
loops.

## Vision model and cost

The default is an inexpensive OpenRouter model that performs well on character
recognition and meme context:

```dotenv
VISION_BASE_URL=https://openrouter.ai/api/v1
VISION_MODEL=google/gemini-3-flash-preview
```

Images and selected video frames are sent to the vision provider. Audio is
transcribed locally, but the resulting transcript is included in the vision
request.

At the time of configuration, Gemini 3 Flash Preview costs $0.50 per 1M input
tokens and $3.00 per 1M output tokens, so a short description usually costs a
fraction of a cent. Configure a spending limit for the OpenRouter key. For a
fully free mode, use `google/gemma-4-26b-a4b-it:free`; it performed worse on
character and meme-context recognition in the sample evaluation. Free
endpoints are also limited to 20 requests per minute and 50 requests per day
for accounts without purchased credits.

Any vision endpoint that supports OpenAI-compatible `chat/completions` and
structured outputs can be selected without code changes:

```dotenv
VISION_API_KEY=...
VISION_BASE_URL=https://openrouter.ai/api/v1
VISION_MODEL=provider/model
```

There is intentionally no automatic fallback to a more expensive model. A
failed request is recorded with the `failed` status and can be queued again.

The indexing prompt uses neutral archival classification. It preserves
profanity, slurs, political material, and dark humor when they are present and
relevant to retrieval, without adding moral commentary or inventing content.

## Maintenance

Queue failed tasks for the next bot start:

```bash
docker compose run --rm mediaarchivistbot \
  python -m archivistbot.cli requeue
```

Recreate every description and embedding, which calls the vision API again:

```bash
docker compose run --rm mediaarchivistbot \
  python -m archivistbot.cli requeue --all
docker compose restart mediaarchivistbot
```

Create missing embeddings without calling the vision API again:

```bash
docker compose run --rm mediaarchivistbot \
  python -m archivistbot.cli reembed
```

This command also recreates vectors produced by another embedding model. A
change in vector dimensionality requires a dedicated Alembic migration for the
`vector(1024)` column.

To measure retrieval quality, create an ignored local file such as
`eval/queries.json`:

```json
[
  {
    "query": "Stilgar as it was written",
    "expected_file_unique_ids": ["telegram-file-unique-id"]
  }
]
```

Then mount it into the maintenance container:

```bash
docker compose run --rm -v "$PWD/eval:/cases:ro" mediaarchivistbot \
  python -m archivistbot.evaluate /cases/queries.json --limit 5
```

The command reports Recall@5, MRR, and the expected meme's rank for each query.

The database schema is managed exclusively by Alembic. Runtime code contains no
`create_all()` call or DDL:

```bash
docker compose run --rm migrate alembic current
docker compose run --rm migrate alembic upgrade head
```

The initial migration can create a clean database or upgrade the legacy
`mediaarchivist.chats/messages/files` schema.

## Backup and restore

Create a compressed PostgreSQL custom-format backup:

```bash
./scripts/backup.sh
```

Backups are written atomically to the ignored `backups/` directory with UTC
timestamps. The script validates the archive with `pg_restore --list` before
publishing it. To use another destination:

```bash
BACKUP_DIR=/srv/mediaarchivist-backups ./scripts/backup.sh
./scripts/backup.sh /srv/mediaarchivist-backups/manual.dump
```

Copy the `.dump` file and this repository to another server, start its database,
and restore the archive:

```bash
./scripts/restore.sh /path/to/mediaarchivist-20260726T180000Z.dump
```

Restore replaces the target database, applies any newer Alembic migrations,
and starts the bot again. Use `--yes` only for unattended, pre-approved
restores. The target should run the same or a newer compatible PostgreSQL major
version and include pgvector.

The backup command is suitable for cron or a systemd timer. For example, this
cron entry creates a daily backup at 03:00 without configuring scheduling in
the repository:

```cron
0 3 * * * cd /opt/MediaArchivistBot && BACKUP_DIR=/srv/mediaarchivist-backups ./scripts/backup.sh >> /var/log/mediaarchivist-backup.log 2>&1
```

Configure retention and off-site copying separately. Backup archives contain
chat metadata, descriptions, transcripts, and embeddings, so store them with
restricted permissions.

## Telegram limitations

- The Bot API does not expose history from before the bot joined. Only new
  messages are indexed.
- The standard Bot API limits downloads to 20 MB per file.
- Deleted source messages cannot be forwarded.
- Protected content is not downloaded or indexed because Telegram prohibits
  forwarding it.
- If the bot is no longer an administrator or Telegram cannot confirm current
  user membership, that chat is excluded from search.

## Development

```bash
pytest
python -m compileall -q archivistbot migrations
docker compose config --quiet
```

Create and apply a migration:

```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

Never commit `.env`. The deleted MinIO experiment contained test credentials;
revoke those credentials even if MinIO is no longer used.

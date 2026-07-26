#!/bin/sh
set -eu

project_dir=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$project_dir"

assume_yes=false
if [ "${1:-}" = "--yes" ]; then
    assume_yes=true
    shift
fi

archive=${1:-}
if [ -z "$archive" ] || [ ! -f "$archive" ]; then
    echo "Usage: $0 [--yes] path/to/backup.dump" >&2
    exit 2
fi

docker compose up -d db >/dev/null
if ! docker compose exec -T db pg_restore --list <"$archive" >/dev/null; then
    echo "Restore aborted: the file is not a valid pg_restore archive." >&2
    exit 1
fi

if [ "$assume_yes" != true ]; then
    printf "This will replace the current MediaArchivist database. Continue? [y/N] "
    read -r answer
    case "$answer" in
        y|Y|yes|YES) ;;
        *) echo "Restore cancelled."; exit 0 ;;
    esac
fi

docker compose stop mediaarchivistbot >/dev/null

docker compose exec -T db sh -c '
    exec pg_restore \
        --username="$POSTGRES_USER" \
        --dbname="$POSTGRES_DB" \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges \
        --exit-on-error \
        --single-transaction
' <"$archive"

docker compose run --rm migrate alembic upgrade head
docker compose up -d mediaarchivistbot

echo "Database restored from: $archive"

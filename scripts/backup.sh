#!/bin/sh
set -eu

umask 077

project_dir=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$project_dir"

timestamp=$(date -u "+%Y%m%dT%H%M%SZ")
backup_dir=${BACKUP_DIR:-"$project_dir/backups"}
output=${1:-"$backup_dir/mediaarchivist-$timestamp.dump"}
temporary="$output.tmp"

mkdir -p "$(dirname "$output")"
rm -f "$temporary"
trap 'rm -f "$temporary"' EXIT HUP INT TERM

docker compose up -d db >/dev/null
docker compose exec -T db sh -c '
    exec pg_dump \
        --username="$POSTGRES_USER" \
        --dbname="$POSTGRES_DB" \
        --format=custom \
        --compress=6 \
        --no-owner \
        --no-privileges
' >"$temporary"

if [ ! -s "$temporary" ]; then
    echo "Backup failed: pg_dump produced an empty archive." >&2
    exit 1
fi

docker compose exec -T db pg_restore --list <"$temporary" >/dev/null
mv "$temporary" "$output"
trap - EXIT HUP INT TERM

echo "Database backup created: $output"

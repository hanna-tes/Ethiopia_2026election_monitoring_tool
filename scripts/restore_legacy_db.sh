#!/usr/bin/env sh
set -eu

if [ "${SKIP_LEGACY_RESTORE:-0}" = "1" ]; then
  echo "Skipping legacy DB restore because SKIP_LEGACY_RESTORE=1."
  exit 0
fi

AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-cfa-bootstrap}"
AWS_REGION="${AWS_REGION:-eu-west-1}"
LEGACY_INSTANCE_ID="${LEGACY_INSTANCE_ID:-i-06c3aa3b62651dc87}"
LEGACY_AVAILABILITY_ZONE="${LEGACY_AVAILABILITY_ZONE:-eu-west-1a}"
LEGACY_HOST="${LEGACY_HOST:-52.49.201.188}"
LEGACY_OS_USER="${LEGACY_OS_USER:-ubuntu}"
LEGACY_DB_NAME="${LEGACY_DB_NAME:-ethiopia_election_db}"
LOCAL_DB_NAME="${LOCAL_DB_NAME:-ethiopia_election_db}"
LOCAL_DB_USER="${LOCAL_DB_USER:-ethiopia_user}"
SSH_PUBLIC_KEY_PATH="${SSH_PUBLIC_KEY_PATH:-$HOME/.ssh/id_ed25519.pub}"
DUMP_PATH="${DUMP_PATH:-/tmp/ethiopia_election_db.dump}"

if [ "${KEEP_LEGACY_DUMP:-0}" != "1" ]; then
  trap 'rm -f "$DUMP_PATH"' EXIT
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required for the default legacy DB restore." >&2
  echo "Set SKIP_LEGACY_RESTORE=1 to run with an empty local database." >&2
  exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
  echo "ssh is required for the default legacy DB restore." >&2
  echo "Set SKIP_LEGACY_RESTORE=1 to run with an empty local database." >&2
  exit 1
fi

if [ ! -f "$SSH_PUBLIC_KEY_PATH" ]; then
  echo "SSH public key not found at $SSH_PUBLIC_KEY_PATH." >&2
  echo "Set SSH_PUBLIC_KEY_PATH=/path/to/key.pub or SKIP_LEGACY_RESTORE=1." >&2
  exit 1
fi

echo "Authorizing temporary SSH key with EC2 Instance Connect..."
aws --profile "$AWS_PROFILE_NAME" ec2-instance-connect send-ssh-public-key \
  --region "$AWS_REGION" \
  --instance-id "$LEGACY_INSTANCE_ID" \
  --availability-zone "$LEGACY_AVAILABILITY_ZONE" \
  --instance-os-user "$LEGACY_OS_USER" \
  --ssh-public-key "file://$SSH_PUBLIC_KEY_PATH" \
  --query 'Success' \
  --output text

echo "Dumping legacy PostgreSQL database from $LEGACY_HOST..."
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
  "$LEGACY_OS_USER@$LEGACY_HOST" \
  "sudo -u postgres pg_dump -Fc '$LEGACY_DB_NAME'" \
  > "$DUMP_PATH"

echo "Restoring dump into local Docker PostgreSQL..."
docker compose stop web worker
docker compose exec -T postgres dropdb -U "$LOCAL_DB_USER" --if-exists "$LOCAL_DB_NAME"
docker compose exec -T postgres createdb -U "$LOCAL_DB_USER" "$LOCAL_DB_NAME"
docker compose exec -T postgres pg_restore -U "$LOCAL_DB_USER" -d "$LOCAL_DB_NAME" --no-owner --no-acl < "$DUMP_PATH"
docker compose run --rm -e DJANGO_COLLECTSTATIC=0 web python manage.py migrate
docker compose run --rm -e DJANGO_COLLECTSTATIC=0 web python manage.py refresh_dashboard_analytics --skip-narratives
docker compose up -d web worker

echo "Legacy database restored into local Docker PostgreSQL."

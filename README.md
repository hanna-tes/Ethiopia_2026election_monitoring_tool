# Ethiopian Election Monitor

Monitoring election-related narratives, hate speech, and coordination patterns for Ethiopia's 2026 election context.

This repository is the Code for Africa maintained version of the app. It is containerized for local development and is being prepared for deployment through the `cfa-iac-pulumi` infrastructure repository.

## Quick Start

The simplest way to run the app locally is Docker Compose. This starts:

- Django web app on port `8000`
- Django-Q worker for background jobs
- PostgreSQL database inside Docker
- MinIO, a local S3-compatible object store, for uploads
- Redis inside Docker for cache and Django-Q background jobs

### Requirements

Install these first:

- Docker Desktop
- Git

### Run Everything

```bash
git clone git@github.com:CodeForAfrica/ethiopian-election-monitor.git
cd ethiopian-election-monitor
docker compose up -d --build
```

Open:

- App: http://127.0.0.1:8000/
- Networks page: http://127.0.0.1:8000/networks/
- Health check: http://127.0.0.1:8000/health/
- MinIO console: http://localhost:9001/

MinIO login:

```text
Username: minioadmin
Password: minioadmin
```

The local S3 bucket is created automatically:

```text
ethiopian-election-monitor-media
```

The first build is large because the image includes PyTorch and Transformers for the AFRO-XLMR detector.

If another local project is already using ports `8000`, `9000`, or `9001`, use alternate host ports:

```bash
APP_PORT=8010 MINIO_API_PORT=9100 MINIO_CONSOLE_PORT=9101 docker compose up -d --build
```

Then open:

- App: http://127.0.0.1:8010/
- MinIO console: http://localhost:9101/

If that alternate app port is also busy, choose another high local port, for example:

```bash
APP_PORT=18021 MINIO_API_PORT=19120 MINIO_CONSOLE_PORT=19121 docker compose up -d --build
```

### Stop Everything

```bash
docker compose down
```

To remove local database and upload data too:

```bash
docker compose down -v
```

## Verify Local Setup

After starting the stack, run:

```bash
curl http://127.0.0.1:8000/health/
```

Expected response:

```json
{"status": "ok"}
```

Check containers:

```bash
docker compose ps
```

You should see `web`, `worker`, `postgres`, `minio`, and `redis` running.

On first startup, the `web` container may show as running before Gunicorn is ready. If the first health check fails, wait a few seconds and retry:

```bash
until curl -fsS http://127.0.0.1:8000/health/; do sleep 3; done
```

## Local Services

| Service | URL or Host | Notes |
| --- | --- | --- |
| Django app | http://127.0.0.1:8000/ | Main application |
| Django-Q worker | `worker` container | Runs queued background jobs |
| Health check | http://127.0.0.1:8000/health/ | Used by local checks and future load balancers |
| MinIO console | http://localhost:9001/ | Browser UI for uploaded files |
| MinIO S3 API | http://localhost:9000/ | S3-compatible API |
| PostgreSQL | `postgres:5432` | Internal Docker network only |
| Redis | `redis:6379` | Internal Docker network only |

PostgreSQL and Redis are intentionally not exposed on host ports. The app connects to them inside the Docker network using `postgres:5432` and `redis:6379`, which is closer to how it will work in ECS/Fargate.

## Configuration

Local Docker Compose sets the required environment variables for you.

Important variables:

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | Django secret key |
| `DEBUG` | Enables local debug mode when set to `1` |
| `ALLOWED_HOSTS` | Comma-separated list of allowed hosts |
| `CSRF_TRUSTED_ORIGINS` | Comma-separated trusted origins |
| `DATABASE_URL` | Optional full database URL |
| `DB_HOST` | PostgreSQL host, for Docker this is `postgres` |
| `DB_PORT` | PostgreSQL port, usually `5432` |
| `DB_NAME` | PostgreSQL database name |
| `DB_USER` | PostgreSQL user |
| `DB_PASSWORD` | PostgreSQL password |
| `CFA_VALKEY_URL` | Redis-compatible URL for CFA managed Valkey in deployment |
| `REDIS_URL` | Redis URL fallback; local Docker points it at `redis://redis:6379/1` |
| `AWS_STORAGE_BUCKET_NAME` | Enables S3-compatible media storage |
| `AWS_S3_ENDPOINT_URL` | S3 API endpoint, MinIO locally |
| `AWS_S3_CUSTOM_DOMAIN` | Public media URL host |
| `AWS_ACCESS_KEY_ID` | S3 or MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | S3 or MinIO secret key |
| `GROQ_API_KEY` | Optional API key for LLM-powered features |
| `GROQ_MODEL` | Optional Groq model override |
| `DATABASE_LOG_HANDLER_ENABLED` | Optional. Set to `1` only if ordinary Python logs should also be mirrored to the database |
| `APPLICATION_LOG_RETENTION_DAYS` | Number of days to retain `ApplicationLog` rows; defaults to `90` |

If `AWS_STORAGE_BUCKET_NAME` is set, uploaded files go to S3-compatible storage. In local Docker, that means MinIO. In AWS, that should be a real S3 bucket provisioned through infrastructure.

If no database environment variables are set, Django falls back to SQLite. For normal development and deployment parity, prefer Docker Compose with PostgreSQL, Redis, and MinIO.

## Upload Storage

Uploads use Django's `default_storage`.

Local behavior:

- Files are saved to MinIO.
- The bucket is created automatically by the `minio-init` service.
- Media URLs use `http://localhost:9000/ethiopian-election-monitor-media/...`.

Production behavior:

- Files should be saved to AWS S3.
- The bucket, IAM permissions, secrets, and app environment variables should be managed through `cfa-iac-pulumi`.

## Redis and Background Jobs

Local Docker Compose runs Redis for Django cache and Django-Q background jobs.
It also starts a `worker` container with `python manage.py qcluster` so queued jobs are processed locally.

Production should use CFA managed Redis/Valkey and inject the Redis-compatible connection URL as:

```text
CFA_VALKEY_URL
```

The app also accepts `REDIS_URL` as a fallback for compatibility.

## Refresh Precomputed Analytics

Some dashboard pages intentionally avoid doing broad scans, clustering, and LLM work during normal page loads. After loading or uploading data, refresh the materialized analytics:

```bash
docker compose exec web python manage.py refresh_dashboard_analytics
```

This command:

- materializes per-post lexicon detections for the `/lexicons/` page
- prepares home-page trend, hashtag, and risk-actor summaries
- prepares narrative summaries for `/narratives/`
- prepares PEP/PIP mention analysis for `/peps/` and `/peps/analysis/`

For a faster local smoke test that skips narrative LLM/clustering work:

```bash
docker compose exec web python manage.py refresh_dashboard_analytics --skip-narratives
```

The real-time text scanner in `/lexicon-management/` still runs live when a user submits text. The refresh command only removes broad page-wide scans from ordinary browsing.

## Optional: Restore Legacy EC2 Data Locally

The normal Docker setup starts with an empty PostgreSQL database. That is enough to confirm the app, worker, Redis, Postgres, and MinIO all start correctly.

For realistic dashboard testing, restore a dump of the legacy EC2 PostgreSQL database into Docker. A CFA engineer with AWS access can create the dump from the legacy host without exposing database passwords:

```bash
aws --profile cfa-bootstrap ec2-instance-connect send-ssh-public-key \
  --region eu-west-1 \
  --instance-id i-06c3aa3b62651dc87 \
  --availability-zone eu-west-1a \
  --instance-os-user ubuntu \
  --ssh-public-key file://$HOME/.ssh/id_ed25519.pub

ssh ubuntu@52.49.201.188 \
  'sudo -u postgres pg_dump -Fc ethiopia_election_db' \
  > /tmp/ethiopia_election_db.dump
```

Restore it into the Docker database:

```bash
docker compose stop web worker
docker compose exec -T postgres dropdb -U ethiopia_user --if-exists ethiopia_election_db
docker compose exec -T postgres createdb -U ethiopia_user ethiopia_election_db
docker compose exec -T postgres pg_restore -U ethiopia_user -d ethiopia_election_db --no-owner --no-acl < /tmp/ethiopia_election_db.dump
docker compose run --rm -e DJANGO_COLLECTSTATIC=0 web python manage.py migrate
docker compose run --rm -e DJANGO_COLLECTSTATIC=0 web python manage.py refresh_dashboard_analytics --skip-narratives
docker compose up -d web worker
```

If you used alternate ports, keep the same `APP_PORT`, `MINIO_API_PORT`, and `MINIO_CONSOLE_PORT` variables on every `docker compose` command.

## LLM or Coding Agent Setup Instructions

If you are asking an LLM or coding agent to set this repo up locally, give it this prompt:

```text
Set up the Ethiopian Election Monitor repo locally.

Use Docker Compose as the primary setup path. Do not create a manual local database unless Docker Compose fails.

Steps:
1. Clone git@github.com:CodeForAfrica/ethiopian-election-monitor.git.
2. Change into the repo.
3. Run docker compose up -d --build.
4. Wait until the web, worker, postgres, minio, and redis services are running.
5. Wait until http://127.0.0.1:8000/health/ returns {"status": "ok"}. Retry every 3 seconds until it is ready.
6. Verify http://127.0.0.1:8000/networks/ returns HTTP 200.
7. Confirm Django is using PostgreSQL, not SQLite.
8. Run docker compose exec web python manage.py refresh_dashboard_analytics --skip-narratives.
9. Verify http://127.0.0.1:8000/lexicons/, http://127.0.0.1:8000/lexicon-management/, and http://127.0.0.1:8000/peps/ return HTTP 200.
10. Confirm uploads use the S3-compatible Django storage backend backed by MinIO.
11. Confirm Django cache uses redis://redis:6379/1.
12. If realistic dashboard data is required, ask a CFA engineer for a legacy Postgres dump or ask them to run the "Optional: Restore Legacy EC2 Data Locally" steps in this README. Do not make Postgres public.

Local URLs:
- App: http://127.0.0.1:8000/
- Networks: http://127.0.0.1:8000/networks/
- Health: http://127.0.0.1:8000/health/
- MinIO console: http://localhost:9001/

MinIO credentials:
- Username: minioadmin
- Password: minioadmin

If port 5432 is already in use on the host, do not change the app database host. The app should use postgres:5432 inside Docker. PostgreSQL does not need to be exposed to the host.
```

Useful verification command for an agent:

```bash
docker compose exec -T web python manage.py shell -c "from django.core.files.base import ContentFile; from django.core.files.storage import default_storage; from django.db import connection; path=default_storage.save('uploads/setup-smoke.txt', ContentFile(b'ok')); print(connection.settings_dict['ENGINE']); print(connection.settings_dict.get('HOST')); print(default_storage.__class__); print(default_storage.open(path, 'rb').read().decode()); print(default_storage.url(path))"
```

Expected signs of success:

```text
django.db.backends.postgresql
postgres
storages.backends.s3.S3Storage
ok
```

Redis verification:

```bash
docker compose exec -T web python manage.py shell -c "from django.conf import settings; from django.core.cache import cache; cache.set('setup-smoke', 'ok', 30); print(settings.REDIS_URL); print(cache.get('setup-smoke'))"
```

Expected:

```text
redis://redis:6379/1
ok
```

## Manual Setup Without Docker

Docker Compose is recommended. Use this only when Docker is unavailable.

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a local `.env` file:

```bash
SECRET_KEY=local-dev-secret
DEBUG=1
ALLOWED_HOSTS=localhost,127.0.0.1
CSRF_TRUSTED_ORIGINS=http://localhost,http://127.0.0.1
```

With only those values, Django will use SQLite and local filesystem media. For parity with deployment, run PostgreSQL and S3-compatible storage and set the database plus `AWS_*` variables listed above.

If you run background jobs or cache features without Docker, also run Redis and set:

```bash
export CFA_VALKEY_URL=redis://127.0.0.1:6379/1
```

Run migrations and start Django:

```bash
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

## Deployment Direction

This app is intended to deploy through Code for Africa infrastructure automation in `cfa-iac-pulumi`.

The production shape should be:

- Django app container on ECS/Fargate
- PostgreSQL via RDS
- Redis via CFA managed Valkey, injected as `CFA_VALKEY_URL`
- Uploads via S3
- Secrets and app configuration managed by infrastructure
- Health check at `/health/`
- Container image built from this repo's `Dockerfile`

Keep local development aligned with that shape by using Docker Compose: app container, PostgreSQL, and S3-compatible object storage.

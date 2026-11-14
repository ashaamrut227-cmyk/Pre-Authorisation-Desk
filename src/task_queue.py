import json
import os
import uuid
from datetime import datetime, timezone

import redis

QUEUE_NAME = "preauth:jobs"
STATUS_PREFIX = "preauth:run:"
RESULT_TTL_SECONDS = int(os.getenv("RUN_RESULT_TTL_SECONDS", "86400"))


def _client():
    return redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
        socket_timeout=15,
        socket_connect_timeout=5,
    )


def enqueue_run(payload: dict) -> str:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary")

    client = _client()
    run_id = str(uuid.uuid4())

    record = {
        "run_id": run_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": payload,
    }

    client.setex(
        STATUS_PREFIX + run_id,
        RESULT_TTL_SECONDS,
        json.dumps(record),
    )

    client.lpush(
        QUEUE_NAME,
        json.dumps({"run_id": run_id, "request": payload}),
    )

    return run_id


def get_run(run_id: str):
    raw = _client().get(STATUS_PREFIX + run_id)
    return json.loads(raw) if raw else None


def update_run(run_id: str, **fields):
    client = _client()
    key = STATUS_PREFIX + run_id
    raw = client.get(key)

    if not raw:
        return

    record = json.loads(raw)
    record.update(fields)

    client.setex(
        key,
        RESULT_TTL_SECONDS,
        json.dumps(record, default=str),
    )


def dequeue_run(timeout: int = 5):
    item = _client().brpop(QUEUE_NAME, timeout=timeout)
    return json.loads(item[1]) if item else None

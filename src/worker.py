import logging
import os
import time
from datetime import datetime, timezone

import redis

from src.agent_metrics import agent_metrics
from src.agents.agent_graph import run_agent_workflow
from src.task_queue import dequeue_run, update_run

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("preauth.worker")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def wait_for_redis():
    client = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://redis:6379/0"),
        socket_connect_timeout=5,
        socket_timeout=5,
    )

    while True:
        try:
            client.ping()
            logger.info("Connected to Redis")
            return
        except redis.RedisError:
            logger.exception("Redis unavailable; retrying in 5 seconds")
            time.sleep(5)


def process_job(job: dict):
    run_id = job.get("run_id")
    request_data = job.get("request")

    if not run_id or not isinstance(request_data, dict):
        logger.error("Invalid job received: %s", job)
        return

    update_run(
        run_id,
        status="running",
        started_at=utc_now(),
    )

    logger.info("Processing run %s", run_id)

    try:
        result = run_agent_workflow(request_data)

        update_run(
            run_id,
            status="completed",
            completed_at=utc_now(),
            result=result,
            human_review_required=(
                result.get("human_review_required", True)
                if isinstance(result, dict)
                else True
            ),
        )

        logger.info("Run %s completed", run_id)

    except Exception as error:
        logger.exception("Run %s failed", run_id)

        update_run(
            run_id,
            status="failed",
            completed_at=utc_now(),
            error=str(error),
            human_review_required=True,
        )


def main():
    wait_for_redis()
    logger.info("Pre-authorisation worker started")

    while True:
        try:
            job = dequeue_run(timeout=5)

            if job:
                process_job(job)

        except redis.RedisError:
            logger.exception("Redis error in worker; retrying shortly")
            time.sleep(3)

        except Exception:
            logger.exception("Unexpected worker error")
            time.sleep(2)


if __name__ == "__main__":
    main()


import os
import threading
import time
from typing import Any

import redis


class AgentMetrics:
    """Shared, thread-safe agent metrics stored in Redis."""

    PREFIX = "preauth:metrics:agent:"

    def __init__(self):
        self.client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/0"),
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        self.lock = threading.Lock()
        self.local_metrics: dict[str, dict[str, Any]] = {}

    def record_execution(
        self,
        agent_name: str,
        duration_seconds: float,
        *,
        success: bool,
        human_review_required: bool = False,
    ) -> None:
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("agent_name must be non-empty.")

        if duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative.")

        key = self.PREFIX + agent_name

        try:
            pipe = self.client.pipeline()
            pipe.hincrby(key, "executions", 1)
            pipe.hincrby(key, "successes" if success else "failures", 1)
            pipe.hincrbyfloat(
                key, "total_duration_seconds", duration_seconds
            )

            if human_review_required:
                pipe.hincrby(key, "human_review_routes", 1)

            pipe.execute()

        except redis.RedisError:
            self._record_local(
                agent_name,
                duration_seconds,
                success,
                human_review_required,
            )

    def _record_local(
        self,
        agent_name,
        duration_seconds,
        success,
        human_review_required,
    ):
        with self.lock:
            metric = self.local_metrics.setdefault(
                agent_name,
                {
                    "executions": 0,
                    "successes": 0,
                    "failures": 0,
                    "total_duration_seconds": 0.0,
                    "human_review_routes": 0,
                },
            )

            metric["executions"] += 1
            metric["successes" if success else "failures"] += 1
            metric["total_duration_seconds"] += duration_seconds

            if human_review_required:
                metric["human_review_routes"] += 1

    def snapshot(self) -> dict[str, dict[str, Any]]:
        result = {}

        try:
            for key in self.client.scan_iter(match=self.PREFIX + "*"):
                agent_name = key[len(self.PREFIX):]
                metric = self.client.hgetall(key)

                executions = int(metric.get("executions", 0))
                total_duration = float(
                    metric.get("total_duration_seconds", 0.0)
                )

                result[agent_name] = {
                    "executions": executions,
                    "successes": int(metric.get("successes", 0)),
                    "failures": int(metric.get("failures", 0)),
                    "total_duration_seconds": total_duration,
                    "human_review_routes": int(
                        metric.get("human_review_routes", 0)
                    ),
                    "average_duration_seconds": (
                        total_duration / executions if executions else 0.0
                    ),
                }

        except redis.RedisError:
            pass

        with self.lock:
            for agent_name, metric in self.local_metrics.items():
                if agent_name not in result:
                    executions = metric["executions"]
                    total_duration = metric["total_duration_seconds"]

                    result[agent_name] = {
                        **metric,
                        "average_duration_seconds": (
                            total_duration / executions
                            if executions else 0.0
                        ),
                    }

        return result

    def reset(self) -> None:
        with self.lock:
            self.local_metrics.clear()

        try:
            keys = list(
                self.client.scan_iter(match=self.PREFIX + "*")
            )
            if keys:
                self.client.delete(*keys)
        except redis.RedisError:
            pass


agent_metrics = AgentMetrics()


def timed_agent_execution(agent_name: str):
    return _TimedAgentExecution(agent_name)


class _TimedAgentExecution:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.started_at = 0.0
        self.human_review_required = False

    def __enter__(self):
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.perf_counter() - self.started_at

        agent_metrics.record_execution(
            self.agent_name,
            duration,
            success=exc_type is None,
            human_review_required=self.human_review_required,
        )

        return False

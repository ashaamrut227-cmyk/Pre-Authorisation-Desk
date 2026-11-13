
import threading
import time
from collections import defaultdict
from typing import Any


class AgentMetrics:
    """Thread-safe in-memory metrics for agent executions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "total_duration_seconds": 0.0,
                "human_review_routes": 0,
            }
        )

    def record_execution(
        self,
        agent_name: str,
        duration_seconds: float,
        *,
        success: bool,
        human_review_required: bool = False,
    ) -> None:
        """Record one completed agent execution."""

        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ValueError("agent_name must be non-empty.")

        if duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative.")

        with self._lock:
            metric = self._metrics[agent_name]
            metric["executions"] += 1
            metric["successes" if success else "failures"] += 1
            metric["total_duration_seconds"] += duration_seconds

            if human_review_required:
                metric["human_review_routes"] += 1

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a safe copy of all recorded metrics."""

        with self._lock:
            result = {}

            for agent_name, metric in self._metrics.items():
                executions = metric["executions"]
                total_duration = metric["total_duration_seconds"]

                result[agent_name] = {
                    **metric,
                    "average_duration_seconds": (
                        total_duration / executions if executions else 0.0
                    ),
                }

            return result

    def reset(self) -> None:
        """Clear metrics. Intended for tests and local development."""

        with self._lock:
            self._metrics.clear()


agent_metrics = AgentMetrics()


def timed_agent_execution(agent_name: str):
    """Return a context manager for measuring an agent's execution."""
    return _TimedAgentExecution(agent_name)


class _TimedAgentExecution:
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.started_at = 0.0
        self.success = False
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


import pytest

from src.agent_metrics import AgentMetrics


def test_records_successful_execution():
    metrics = AgentMetrics()

    metrics.record_execution(
        "intake_agent",
        0.5,
        success=True,
        human_review_required=True,
    )

    result = metrics.snapshot()["intake_agent"]

    assert result["executions"] == 1
    assert result["successes"] == 1
    assert result["failures"] == 0
    assert result["human_review_routes"] == 1
    assert result["average_duration_seconds"] == pytest.approx(0.5)


def test_records_failed_execution():
    metrics = AgentMetrics()

    metrics.record_execution(
        "policy_agent",
        1.5,
        success=False,
    )

    result = metrics.snapshot()["policy_agent"]

    assert result["executions"] == 1
    assert result["failures"] == 1
    assert result["successes"] == 0


def test_calculates_average_duration():
    metrics = AgentMetrics()

    metrics.record_execution("intake_agent", 1.0, success=True)
    metrics.record_execution("intake_agent", 3.0, success=True)

    result = metrics.snapshot()["intake_agent"]

    assert result["executions"] == 2
    assert result["average_duration_seconds"] == pytest.approx(2.0)


def test_rejects_invalid_duration():
    metrics = AgentMetrics()

    with pytest.raises(ValueError):
        metrics.record_execution(
            "intake_agent",
            -1.0,
            success=True,
        )


def test_reset_clears_metrics():
    metrics = AgentMetrics()
    metrics.record_execution("intake_agent", 0.2, success=True)

    metrics.reset()

    assert metrics.snapshot() == {}

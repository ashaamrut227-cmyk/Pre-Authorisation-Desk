
from src.temporal_verifier import verify_temporal_applicability


def test_waiting_period_not_elapsed():
    result = verify_temporal_applicability(
        policy_start_date="2025-01-01",
        admission_date="2025-12-01",
        waiting_period_months=12,
    )

    assert result["status"] == "not_elapsed"
    assert result["human_review_required"] is True


def test_waiting_period_elapsed():
    result = verify_temporal_applicability(
        policy_start_date="2025-01-01",
        admission_date="2026-01-01",
        waiting_period_months=12,
    )

    assert result["status"] == "elapsed"


def test_admission_before_policy_start():
    result = verify_temporal_applicability(
        policy_start_date="2025-01-01",
        admission_date="2024-12-01",
        waiting_period_months=12,
    )

    assert result["status"] == "contradicted"


def test_invalid_date():
    result = verify_temporal_applicability(
        policy_start_date="invalid",
        admission_date="2026-01-01",
        waiting_period_months=12,
    )

    assert result["status"] == "unverified"


from datetime import date
from typing import Any


def verify_temporal_applicability(
    policy_start_date: str,
    admission_date: str,
    waiting_period_months: int,
) -> dict[str, Any]:
    """
    Check whether a waiting period has elapsed by admission date.

    This is a preliminary date check, not a final coverage decision.
    """

    try:
        start = date.fromisoformat(policy_start_date)
        admission = date.fromisoformat(admission_date)
    except (TypeError, ValueError):
        return {
            "status": "unverified",
            "reason": "Invalid policy or admission date.",
            "human_review_required": True,
        }

    if admission < start:
        return {
            "status": "contradicted",
            "reason": "Admission date is before policy start date.",
            "human_review_required": True,
        }

    if (
        not isinstance(waiting_period_months, int)
        or isinstance(waiting_period_months, bool)
        or waiting_period_months < 0
    ):
        return {
            "status": "unverified",
            "reason": "Waiting period must be a non-negative integer.",
            "human_review_required": True,
        }

    month_index = start.year * 12 + start.month - 1
    target_index = month_index + waiting_period_months
    target_year, target_month_zero = divmod(target_index, 12)

    target_month = target_month_zero + 1
    import calendar
    target_day = min(
        start.day,
        calendar.monthrange(target_year, target_month)[1],
    )
    waiting_period_end = date(
        target_year, target_month, target_day
    )

    elapsed = admission >= waiting_period_end

    return {
        "status": "elapsed" if elapsed else "not_elapsed",
        "policy_start_date": start.isoformat(),
        "admission_date": admission.isoformat(),
        "waiting_period_months": waiting_period_months,
        "waiting_period_end_date": waiting_period_end.isoformat(),
        "human_review_required": True,
        "reason": (
            "Waiting period has elapsed."
            if elapsed
            else "Waiting period has not elapsed."
        ),
    }

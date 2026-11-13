
from typing import Any


ADMISSION_TYPES = {
    "planned_surgical",
    "day_care",
    "emergency",
    "maternity",
}


def classify_admission(
    procedure_name: str,
    admission_type: str,
) -> dict[str, Any]:
    """Identify admission-type clues without overriding clinical input."""

    if not isinstance(procedure_name, str) or not procedure_name.strip():
        raise ValueError("procedure_name must be a non-empty string.")

    if admission_type not in ADMISSION_TYPES:
        raise ValueError(
            f"admission_type must be one of {sorted(ADMISSION_TYPES)}."
        )

    procedure = procedure_name.strip().lower()

    keyword_groups = {
        "emergency": (
            "emergency",
            "accident",
            "urgent",
            "trauma",
        ),
        "maternity": (
            "maternity",
            "delivery",
            "childbirth",
            "caesarean",
            "cesarean",
        ),
        "day_care": (
            "day care",
            "cataract",
            "dialysis",
            "chemotherapy",
        ),
        "planned_surgical": (
            "surgery",
            "surgical",
            "operation",
        ),
    }

    detected_clues = [
        category
        for category, keywords in keyword_groups.items()
        if any(keyword in procedure for keyword in keywords)
    ]

    # Day-care treatment can also involve a planned surgical procedure.
    # These two clues alone do not establish a classification conflict.
    overlapping_clues = {"day_care", "planned_surgical"}

    only_overlapping_pair = (
        len(detected_clues) > 1
        and set(detected_clues) == overlapping_clues
    )

    # Multiple incompatible clues require clarification.
    # A single clue that disagrees with the supplied type also requires review.
    classification_conflict = (
        (len(detected_clues) > 1 and not only_overlapping_pair)
        or (
            len(detected_clues) == 1
            and admission_type not in detected_clues
        )
    )

    return {
        "provided_admission_type": admission_type,
        "procedure_name": procedure.strip(),
        "detected_clues": detected_clues,
        "classification_conflict": classification_conflict,
        "human_review_required": True,
    }


def extract_request_entities(
    request_data: dict[str, Any],
) -> dict[str, Any]:
    """Extract structured fields from a validated request."""

    if not isinstance(request_data, dict):
        raise ValueError("request_data must be a dictionary.")

    procedure = request_data.get("procedure_name", "")
    amount = request_data.get("requested_amount")
    policy_limit = request_data.get("policy_limit")

    if not isinstance(procedure, str) or not procedure.strip():
        raise ValueError("procedure_name is required.")

    amount_value = None
    limit_value = None

    try:
        if amount is not None:
            amount_value = float(amount)

        if policy_limit is not None:
            limit_value = float(policy_limit)

    except (TypeError, ValueError) as error:
        raise ValueError(
            "Requested amount and policy limit must be numeric."
        ) from error

    if amount_value is not None and amount_value <= 0:
        raise ValueError("requested_amount must be positive.")

    if limit_value is not None and limit_value <= 0:
        raise ValueError("policy_limit must be positive.")

    return {
        "request_id": request_data.get("request_id"),
        "policy_id": request_data.get("policy_id"),
        "procedure_name": procedure.strip(),
        "admission_date": request_data.get("admission_date"),
        "admission_type": request_data.get("admission_type"),
        "requested_amount": amount_value,
        "policy_limit": limit_value,
        "required_documents_complete": request_data.get(
            "required_documents_complete"
        ),
        "room_category": request_data.get("room_category"),
        "amount_exceeds_policy_limit": (
            amount_value is not None
            and limit_value is not None
            and amount_value > limit_value
        ),
    }


from typing import Any


VALID_ROUTES = {
    "approve-recommended",
    "query-to-hospital",
    "medical-review",
}


def route_pre_authorisation(
    request_data: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    """
    Recommend a workflow route using document completeness
    and policy-evidence verification.

    This function does not make a final insurance decision.
    """

    if not isinstance(request_data, dict):
        raise ValueError("request_data must be a dictionary.")

    if not isinstance(verification, dict):
        raise ValueError("verification must be a dictionary.")

    documents_complete = request_data.get(
        "required_documents_complete", False
    )

    if not isinstance(documents_complete, bool):
        raise ValueError(
            "required_documents_complete must be a boolean."
        )

    status = verification.get("status", "unverified")

    if status not in {"supported", "contradicted", "unverified"}:
        status = "unverified"

    if not documents_complete:
        route = "query-to-hospital"
        reason = (
            "Required documents are incomplete. "
            "Request the missing documents before proceeding."
        )

    elif status != "supported":
        route = "medical-review"
        reason = (
            "Policy evidence does not sufficiently support the claim "
            "or conflicts with it. Human review is required."
        )

    else:
        route = "approve-recommended"
        reason = (
            "The supplied evidence supports the recognized claim facts "
            "and the request indicates that required documents are "
            "complete. This is a recommendation only, not approval."
        )

    return {
        "route": route,
        "reason": reason,
        "human_review_required": True,
        "final_coverage_decision": None,
    }

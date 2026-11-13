
from typing import Any

from src.agents.tools import intake_tool


def run_intake_agent(
    request_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Intake Agent:
    Extract request information and determine whether policy review
    can proceed safely.
    """

    if not isinstance(request_data, dict):
        return {
            "agent": "intake_agent",
            "status": "quarantined",
            "reason": "The request must be a dictionary.",
            "handoff_to": None,
            "data": None,
        }

    try:
        intake_result = intake_tool(request_data)

    except (ValueError, TypeError, KeyError) as error:
        return {
            "agent": "intake_agent",
            "status": "quarantined",
            "reason": str(error),
            "handoff_to": None,
            "data": None,
        }

    entities = intake_result["entities"]
    classification = intake_result["classification"]

    if not entities.get("required_documents_complete", False):
        return {
            "agent": "intake_agent",
            "status": "needs_information",
            "reason": "Required documents are incomplete.",
            "handoff_to": "query-to-hospital",
            "data": intake_result,
        }

    if classification["classification_conflict"]:
        return {
            "agent": "intake_agent",
            "status": "needs_review",
            "reason": (
                "Admission-type clues are conflicting or ambiguous."
            ),
            "handoff_to": "medical-review",
            "data": intake_result,
        }

    return {
        "agent": "intake_agent",
        "status": "ready_for_policy_review",
        "reason": "Intake checks completed.",
        "handoff_to": "policy_agent",
        "data": intake_result,
    }

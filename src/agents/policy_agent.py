
import logging
import re
from typing import Any

from src.agents.tools import (
    retrieve_policy_tool,
    route_request_tool,
    verify_policy_claim_tool,
)
from src.temporal_verifier import verify_temporal_applicability


logger = logging.getLogger(__name__)


def _find_waiting_period_months(
    procedure: str,
    evidence: list[dict[str, Any]],
) -> int | None:
    """Extract a procedure-specific waiting period from policy sections."""

    ignored_words = {
        "surgery",
        "procedure",
        "treatment",
        "operation",
        "medical",
        "planned",
        "surgical",
    }

    terms = {
        word.lower()
        for word in re.findall(r"[a-zA-Z]+", procedure)
        if len(word) >= 4 and word.lower() not in ignored_words
    }

    if not terms:
        return None

    durations = set()

    for item in evidence:
        text = item.get("text", "")

        if not isinstance(text, str):
            continue

        lines = text.splitlines()

        for index, line in enumerate(lines):
            line_lower = line.lower()

            if not re.search(r"\bwaiting periods?\b", line_lower):
                continue

            matches = re.findall(
                r"\b(\d+)\s*(months?|years?)\b",
                line_lower,
            )

            if not matches:
                continue

            # Collect the heading and its associated condition list.
            section_lines = [line]

            for following_line in lines[index + 1:index + 26]:
                if re.search(
                    r"\bwaiting periods?\b",
                    following_line.lower(),
                ):
                    break

                section_lines.append(following_line)

            section_text = " ".join(section_lines).lower()

            condition_found = any(
                re.search(
                    rf"\b{re.escape(term)}\b",
                    section_text,
                )
                for term in terms
            )

            if not condition_found:
                continue

            for number, unit in matches:
                value = int(number)

                months = (
                    value * 12
                    if unit.startswith("year")
                    else value
                )

                durations.add(months)

    # Do not guess when multiple waiting periods are found.
    return next(iter(durations)) if len(durations) == 1 else None


def run_policy_agent(
    request_data: dict[str, Any],
) -> dict[str, Any]:
    """Retrieve policy evidence, verify claims, and recommend a route."""

    if not isinstance(request_data, dict):
        return {
            "agent": "policy_agent",
            "status": "error",
            "reason": "The request must be a dictionary.",
            "evidence": [],
            "verification": None,
            "temporal_verification": None,
            "routing": None,
            "human_review_required": True,
            "final_coverage_decision": None,
        }

    procedure = request_data.get("procedure_name")

    if not isinstance(procedure, str) or not procedure.strip():
        return {
            "agent": "policy_agent",
            "status": "needs_information",
            "reason": "A procedure name is required.",
            "evidence": [],
            "verification": None,
            "temporal_verification": None,
            "routing": None,
            "human_review_required": True,
            "final_coverage_decision": None,
        }

    question = (
        f"What policy provisions apply to {procedure}, including "
        "waiting periods, exclusions, and limits?"
    )

    claim = f"The policy provides coverage for {procedure}."

    try:
        # Step 1: Retrieve relevant policy evidence.
        evidence = retrieve_policy_tool(question, top_k=8)

        # Step 2: Verify the coverage claim against policy evidence.
        verification = verify_policy_claim_tool(claim, evidence)

        # Step 3: Extract the condition-specific waiting period.
        waiting_period_months = _find_waiting_period_months(
            procedure,
            evidence,
        )

        # Step 4: Check whether the waiting period has elapsed.
        if waiting_period_months is not None:
            temporal_verification = verify_temporal_applicability(
                policy_start_date=request_data.get("policy_start_date"),
                admission_date=request_data.get("admission_date"),
                waiting_period_months=waiting_period_months,
            )
        else:
            temporal_verification = {
                "status": "unverified",
                "reason": (
                    "No unambiguous procedure-specific waiting period "
                    "was found in the retrieved policy evidence."
                ),
                "human_review_required": True,
            }

        # Step 5: Recommend a workflow route.
        routing = route_request_tool(
            request_data,
            verification,
        )

    except Exception:
        # Log the complete traceback for debugging.
        logger.exception(
            "Policy evaluation failed for request_id=%s",
            request_data.get("request_id"),
        )

        return {
            "agent": "policy_agent",
            "status": "error",
            "reason": (
                "Policy evaluation failed. Human review is required."
            ),
            "evidence": [],
            "verification": None,
            "temporal_verification": None,
            "routing": {
                "route": "medical-review",
                "reason": "Policy evaluation could not be completed.",
                "human_review_required": True,
                "final_coverage_decision": None,
            },
            "human_review_required": True,
            "final_coverage_decision": None,
        }

    # An unelapsed or contradicted waiting period requires medical review.
    if temporal_verification["status"] in {
        "not_elapsed",
        "contradicted",
    }:
        routing = {
            "route": "medical-review",
            "reason": temporal_verification["reason"],
            "human_review_required": True,
            "final_coverage_decision": None,
        }

    # Check for admission classification conflicts.
    from src.agents.tools import intake_tool

    try:
        intake_result = intake_tool(request_data)
        classification = intake_result["classification"]

        if classification["classification_conflict"]:
            routing = {
                "route": "medical-review",
                "reason": (
                    "Admission-type clues are conflicting or ambiguous."
                ),
                "human_review_required": True,
                "final_coverage_decision": None,
            }

    except (ValueError, TypeError, KeyError):
        logger.exception("Intake classification validation failed.")

        routing = {
            "route": "medical-review",
            "reason": "Intake details could not be validated.",
            "human_review_required": True,
            "final_coverage_decision": None,
        }

    # Step 6: Return the complete policy-agent result.
    return {
        "agent": "policy_agent",
        "status": verification["status"],
        "reason": (
            "Policy evidence retrieval and verification completed."
        ),
        "evidence": evidence,
        "verification": verification,
        "temporal_verification": temporal_verification,
        "routing": routing,
        "human_review_required": True,
        "final_coverage_decision": None,
    }

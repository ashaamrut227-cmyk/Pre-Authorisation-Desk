
from src.agents.intake_agent import run_intake_agent


def sample_request():
    return {
        "request_id": "REQ-1001",
        "policy_id": "POL-1001",
        "procedure_name": "General consultation",
        "admission_date": "2026-10-15",
        "admission_type": "planned_surgical",
        "requested_amount": 35000,
        "policy_limit": 100000,
        "required_documents_complete": True,
        "room_category": "Standard",
    }


def test_intake_agent_hands_off_valid_request():
    result = run_intake_agent(sample_request())

    assert result["status"] == "ready_for_policy_review"
    assert result["handoff_to"] == "policy_agent"


def test_intake_agent_routes_missing_documents():
    request = sample_request()
    request["required_documents_complete"] = False

    result = run_intake_agent(request)

    assert result["status"] == "needs_information"
    assert result["handoff_to"] == "query-to-hospital"


def test_intake_agent_allows_overlapping_cataract_clues():
    request = sample_request()
    request["procedure_name"] = "Cataract surgery"

    result = run_intake_agent(request)

    assert result["status"] == "ready_for_policy_review"
    assert result["handoff_to"] == "policy_agent"


def test_intake_agent_quarantines_invalid_request():
    result = run_intake_agent({})

    assert result["status"] == "quarantined"
    assert result["handoff_to"] is None

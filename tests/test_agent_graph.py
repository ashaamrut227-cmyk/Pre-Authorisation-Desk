
from src.agents.agent_graph import run_agent_workflow


def sample_request():
    return {
        "request_id": "REQ-2001",
        "policy_id": "POL-2001",
        "procedure_name": "General consultation",
        "admission_date": "2026-10-15",
        "admission_type": "planned_surgical",
        "requested_amount": 35000,
        "policy_limit": 100000,
        "required_documents_complete": True,
        "room_category": "Standard",
    }


def test_graph_hands_valid_request_to_policy_agent():
    result = run_agent_workflow(sample_request())

    assert result["intake"]["status"] == "ready_for_policy_review"
    assert result["policy_review"] is not None
    assert result["human_review_required"] is True
    assert result["final_coverage_decision"] is None


def test_graph_stops_when_documents_are_missing():
    request = sample_request()
    request["required_documents_complete"] = False

    result = run_agent_workflow(request)

    assert result["status"] == "needs_information"
    assert result["route"] == "query-to-hospital"
    assert result["policy_review"] is None


def test_graph_hands_cataract_request_to_policy_agent():
    request = sample_request()
    request["procedure_name"] = "Cataract surgery"

    result = run_agent_workflow(request)

    assert result["intake"]["status"] == "ready_for_policy_review"
    assert result["policy_review"] is not None
    assert result["human_review_required"] is True
    assert result["final_coverage_decision"] is None

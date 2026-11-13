
from src.agents import policy_agent


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


def test_policy_agent_returns_evidence_and_route(monkeypatch):
    monkeypatch.setattr(
        policy_agent,
        "retrieve_policy_tool",
        lambda question, top_k=8: [{
            "page_number": 2,
            "chunk_id": "page-2-part-1",
            "text": "Example policy evidence.",
        }],
    )
    monkeypatch.setattr(
        policy_agent,
        "verify_policy_claim_tool",
        lambda claim, evidence: {
            "status": "supported",
            "verified": True,
            "reason": "Mocked verification.",
        },
    )
    monkeypatch.setattr(
        policy_agent,
        "route_request_tool",
        lambda request_data, verification: {
            "route": "approve-recommended",
            "reason": "Mocked route.",
            "human_review_required": True,
            "final_coverage_decision": None,
        },
    )

    result = policy_agent.run_policy_agent(sample_request())

    assert result["agent"] == "policy_agent"
    assert result["status"] == "supported"
    assert result["evidence"][0]["page_number"] == 2
    assert result["routing"]["route"] == "approve-recommended"
    assert result["human_review_required"] is True
    assert result["final_coverage_decision"] is None


def test_policy_agent_handles_missing_procedure():
    result = policy_agent.run_policy_agent({
        "request_id": "REQ-1002",
    })

    assert result["status"] == "needs_information"
    assert result["human_review_required"] is True


def test_policy_agent_handles_retrieval_failure(monkeypatch):
    def fail_retrieval(question, top_k=8):
        raise RuntimeError("Simulated retrieval failure")

    monkeypatch.setattr(
        policy_agent,
        "retrieve_policy_tool",
        fail_retrieval,
    )

    result = policy_agent.run_policy_agent(sample_request())

    assert result["status"] == "error"
    assert result["routing"]["route"] == "medical-review"
    assert result["final_coverage_decision"] is None
def test_policy_agent_checks_waiting_period(monkeypatch):
    request = sample_request()
    request["policy_start_date"] = "2024-01-01"
    request["admission_date"] = "2026-11-10"
    request["procedure_name"] = "Cataract surgery"

    evidence = [{
        "page_number": 12,
        "chunk_id": "page-12-part-4",
        "text": (
            "A waiting period of 24 months applies "
            "to cataract treatment."
        ),
    }]

    monkeypatch.setattr(
        policy_agent,
        "retrieve_policy_tool",
        lambda question, top_k=8: evidence,
    )
    monkeypatch.setattr(
        policy_agent,
        "verify_policy_claim_tool",
        lambda claim, evidence: {
            "status": "unverified",
            "verified": False,
            "reason": "Coverage requires further review.",
        },
    )
    monkeypatch.setattr(
        policy_agent,
        "route_request_tool",
        lambda request_data, verification: {
            "route": "medical-review",
            "reason": "Human review required.",
            "human_review_required": True,
            "final_coverage_decision": None,
        },
    )
    monkeypatch.setattr(
        "src.agents.tools.intake_tool",
        lambda request_data: {
            "classification": {"classification_conflict": False}
        },
    )

    result = policy_agent.run_policy_agent(request)

    assert result["temporal_verification"]["status"] == "elapsed"
    assert result["temporal_verification"]["waiting_period_months"] == 24
    assert result["routing"]["route"] == "medical-review"
    assert result["final_coverage_decision"] is None


def test_policy_agent_ignores_unrelated_waiting_period():
    evidence = [{
        "text": (
            "A waiting period of 24 months applies "
            "to dental implants."
        ),
    }]

    result = policy_agent._find_waiting_period_months(
        "Cataract surgery",
        evidence,
    )

    assert result is None


def test_policy_agent_rejects_conflicting_waiting_periods():
    evidence = [{
        "text": (
            "A waiting period of 24 months applies to cataract treatment. "
            "A waiting period of 36 months also applies to cataract treatment."
        ),
    }]

    result = policy_agent._find_waiting_period_months(
        "Cataract surgery",
        evidence,
    )

    assert result is None
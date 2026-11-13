
import pytest

from src.decision_router import route_pre_authorisation


def test_supported_claim_recommends_review():
    result = route_pre_authorisation(
        {"required_documents_complete": True},
        {"status": "supported"},
    )

    assert result["route"] == "approve-recommended"
    assert result["human_review_required"] is True
    assert result["final_coverage_decision"] is None


def test_missing_documents_routes_to_hospital_query():
    result = route_pre_authorisation(
        {"required_documents_complete": False},
        {"status": "supported"},
    )

    assert result["route"] == "query-to-hospital"


@pytest.mark.parametrize(
    "status",
    ["contradicted", "unverified"],
)
def test_unsupported_evidence_routes_to_medical_review(status):
    result = route_pre_authorisation(
        {"required_documents_complete": True},
        {"status": status},
    )

    assert result["route"] == "medical-review"


def test_invalid_document_flag_is_rejected():
    with pytest.raises(ValueError):
        route_pre_authorisation(
            {"required_documents_complete": "yes"},
            {"status": "supported"},
        )


def test_invalid_verification_status_routes_to_review():
    result = route_pre_authorisation(
        {"required_documents_complete": True},
        {"status": "unknown"},
    )

    assert result["route"] == "medical-review"

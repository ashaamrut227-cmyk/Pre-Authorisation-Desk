
import pytest

from src.admission_classifier import (
    classify_admission,
    extract_request_entities,
)


def test_classifies_cataract_keyword_as_day_care_clue():
    result = classify_admission(
        "Cataract surgery",
        "planned_surgical",
    )

    assert "day_care" in result["detected_clues"]
    assert "planned_surgical" in result["detected_clues"]
    assert result["classification_conflict"] is False
    assert result["human_review_required"] is True


def test_rejects_invalid_admission_type():
    with pytest.raises(ValueError):
        classify_admission("Cataract surgery", "unknown")


def test_extracts_request_entities():
    result = extract_request_entities({
        "request_id": "REQ-1001",
        "policy_id": "POL-1001",
        "procedure_name": "Cataract surgery",
        "admission_date": "2026-10-15",
        "admission_type": "planned_surgical",
        "requested_amount": 35000,
        "policy_limit": 100000,
        "required_documents_complete": True,
        "room_category": "Standard",
    })

    assert result["procedure_name"] == "Cataract surgery"
    assert result["requested_amount"] == 35000.0
    assert result["amount_exceeds_policy_limit"] is False


def test_detects_amount_above_policy_limit():
    result = extract_request_entities({
        "procedure_name": "Example procedure",
        "requested_amount": 120000,
        "policy_limit": 100000,
    })

    assert result["amount_exceeds_policy_limit"] is True


def test_rejects_non_positive_amount():
    with pytest.raises(ValueError):
        extract_request_entities({
            "procedure_name": "Example procedure",
            "requested_amount": 0,
        })

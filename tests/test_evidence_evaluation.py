import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_verifier import verify_evidence


def make_evidence(text, page_number=12):
    return [
        {
            "document_title": "EquiCover Health Policy Wording",
            "source_file": "policy_wording.pdf",
            "page_number": page_number,
            "chunk_id": "test-chunk-001",
            "text": text,
        }
    ]


def test_empty_evidence_returns_unverified():
    result = verify_evidence(
        "The policy provides coverage for cataract.",
        [],
    )
    assert result["status"] == "unverified"
    assert result["verified"] is False
    assert result["matched_passages"] == []


def test_cataract_waiting_period_with_matching_duration():
    evidence = make_evidence(
        "Cataract surgery is subject to a waiting period of 24 months."
    )
    result = verify_evidence(
        "The waiting period for cataract is 24 months.",
        evidence,
    )
    assert result["status"] == "supported"
    assert result["verified"] is True
    assert result["matched_passages"]
    assert result["matched_passages"][0]["page_number"] == 12


def test_cataract_waiting_period_with_conflicting_duration():
    evidence = make_evidence(
        "Cataract surgery is subject to a waiting period of 24 months."
    )
    result = verify_evidence(
        "The waiting period for cataract is 12 months.",
        evidence,
    )
    assert result["status"] == "contradicted"
    assert result["verified"] is False


def test_no_waiting_period_claim_is_not_supported_by_silence():
    evidence = make_evidence(
        "The policy describes cataract treatment benefits."
    )
    result = verify_evidence(
        "There is no waiting period for cataract.",
        evidence,
    )
    assert result["status"] == "unverified"
    assert result["verified"] is False


def test_guaranteed_approval_claim_is_not_supported():
    evidence = make_evidence(
        "Cataract treatment is subject to policy terms and conditions."
    )
    result = verify_evidence(
        "Cataract treatment is guaranteed approval.",
        evidence,
    )
    assert result["status"] == "unverified"
    assert result["verified"] is False


def test_unsupported_condition_is_unverified():
    evidence = make_evidence(
        "The policy describes cataract waiting periods."
    )
    result = verify_evidence(
        "The policy provides coverage for dental implants.",
        evidence,
    )
    assert result["status"] == "unverified"
    assert result["verified"] is False
    assert result["unmatched_terms"]


def test_waiting_period_must_match_the_same_condition():
    evidence = make_evidence(
        "Cataract surgery is subject to a waiting period of 24 months. "
        "Dental implants are subject to a waiting period of 12 months."
    )
    result = verify_evidence(
        "The waiting period for cataract is 12 months.",
        evidence,
    )
    assert result["status"] == "contradicted"


def test_evidence_page_and_chunk_are_preserved():
    evidence = [
        {
            "document_title": "Policy wording",
            "source_file": "policy_wording.pdf",
            "page_number": 18,
            "chunk_id": "chunk-018",
            "text": "Cataract surgery has a waiting period of 24 months.",
        }
    ]
    result = verify_evidence(
        "The waiting period for cataract is 24 months.",
        evidence,
    )
    assert result["matched_passages"]
    assert result["matched_passages"][0]["page_number"] == 18
    assert result["matched_passages"][0]["chunk_id"] == "chunk-018"


def test_empty_claim_is_rejected():
    with pytest.raises(ValueError):
        verify_evidence("", make_evidence("Policy wording."))


def test_unrelated_waiting_period_does_not_verify_cataract():
    evidence = make_evidence(
        "Dental implants have a waiting period of 12 months."
    )
    result = verify_evidence(
        "The waiting period for cataract is 12 months.",
        evidence,
    )
    assert result["status"] == "unverified"
    assert result["verified"] is False


import pytest
from pydantic import ValidationError

from app.schemas import PreAuthRequest


def valid_request():
    return {
        "request_id": "REQ-001",
        "policy_id": "POL-001",
        "policy_start_date": "2025-11-01",
        "admission_date": "2026-10-20",
        "admission_type": "planned_surgical",
        "procedure_name": "Cataract surgery",
        "requested_amount": 45000,
        "policy_limit": 100000,
        "required_documents_complete": True,
        "room_category": "Standard",
    }


def test_valid_request_is_accepted():
    request = PreAuthRequest(**valid_request())

    assert request.request_id == "REQ-001"
    assert request.admission_type.value == "planned_surgical"


def test_negative_requested_amount_is_rejected():
    data = valid_request()
    data["requested_amount"] = -100

    with pytest.raises(ValidationError):
        PreAuthRequest(**data)


def test_invalid_admission_type_is_rejected():
    data = valid_request()
    data["admission_type"] = "unknown_type"

    with pytest.raises(ValidationError):
        PreAuthRequest(**data)


def test_admission_before_policy_start_is_rejected():
    data = valid_request()
    data["admission_date"] = "2025-10-01"

    with pytest.raises(ValidationError):
        PreAuthRequest(**data)

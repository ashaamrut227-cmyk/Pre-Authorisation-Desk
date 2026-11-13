
from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class AdmissionType(str, Enum):
    PLANNED_SURGICAL = "planned_surgical"
    DAY_CARE = "day_care"
    EMERGENCY = "emergency"
    MATERNITY = "maternity"


class PreAuthRequest(BaseModel):
    request_id: str = Field(min_length=3, max_length=50)
    policy_id: str = Field(min_length=3, max_length=50)
    policy_start_date: date
    admission_date: date
    admission_type: AdmissionType
    procedure_name: str = Field(min_length=2, max_length=150)
    requested_amount: float = Field(gt=0, le=10_000_000)
    policy_limit: float = Field(gt=0, le=100_000_000)
    required_documents_complete: bool
    room_category: str = Field(min_length=2, max_length=50)

    @model_validator(mode="after")
    def validate_dates(self):
        if self.admission_date < self.policy_start_date:
            raise ValueError(
                "Admission date cannot be before policy start date"
            )
        return self

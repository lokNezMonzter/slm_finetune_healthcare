from pydantic import BaseModel, Field, ValidationError
from typing import Literal, List

from parser import parse_json


class Relationship(BaseModel):
    subject: str
    predicate: str
    object: str
    polarity: Literal["positive", "negative"]
    certainty: Literal["confirmed", "suspected", "hedged"]
    evidence: str


class ClinicalExtraction(BaseModel):
    summary: str = Field(..., min_length=32)
    clinical_reasoning: str = Field(..., min_length=32, max_length=1024)
    relationships: List[Relationship]
    keywords: List[str]


def validate_record(data, record_id):
    """
    Returns (is_valid, cleaned_data, status_message).
    """
    parsed = parse_json(data) if isinstance(data, str) else data
    if parsed is None:
        return False, None, "unrecoverable_json"
    
    try:
        validated = ClinicalExtraction(**parsed)
        return True, validated.model_dump(), "valid"   
    except ValidationError as e:
        error_msg = "; ".join([f"{err['loc']}: {err['msg']}" for err in e.errors()])
        return False, parsed, f"schema_error: {error_msg}"
    
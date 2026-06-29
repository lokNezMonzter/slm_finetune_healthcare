from pydantic import BaseModel, Field, ValidationError
from typing import Literal, List
from parser import parse_json

class ProtocolRelationship(BaseModel):
    source: str
    type: Literal["HAS_INDICATION", "TESTS_INTERVENTION", "USES_CONTROL", "MEASURES_ENDPOINT","TARGETS_BIOMARKER", "REQUIRES_CRITERION", "EXCLUDES_CRITERION"]
    target: str

class ClinicalExtraction(BaseModel):
    protocol: str = Field(..., description="Official Trial Title")
    reasoning: str = Field(..., min_length=32, description="Chain of thought reasoning trace")
    relationships: List[ProtocolRelationship]

def validate_record(data, record_id):
    parsed = parse_json(data) if isinstance(data, str) else data
    if parsed is None:
        return False, None, "unrecoverable_json"
    
    try:
        validated = ClinicalExtraction(**parsed)
        return True, validated.model_dump(), "valid"   
    except ValidationError as e:
        error_msg = "; ".join([f"{err['loc']}: {err['msg']}" for err in e.errors()])
        return False, parsed, f"schema_error: {error_msg}"
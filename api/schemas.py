"""
Pydantic v2 schemas for API requests, responses, and validation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class MedicationItemResponse(BaseModel):
    """Details of a single recognized medication item."""
    line_index: int
    raw_ocr_text: str
    verified_drug_name: str
    original_candidate_name: str
    dosage_strength: Optional[str] = None
    dosage_form: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instructions: Optional[str] = None
    route: Optional[str] = None
    ocr_confidence: float
    lexicon_similarity: float
    final_confidence: float
    needs_review: bool
    review_reasons: List[str] = []
    bounding_box: Optional[Dict[str, int]] = None


class PrescriptionRecognitionResponse(BaseModel):
    """Response payload returned from prescription recognition endpoint."""
    prescription_id: str
    timestamp: str
    total_medications_detected: int
    overall_confidence: float
    safety_status: str
    needs_manual_review: bool
    deskew_angle_corrected: float
    medications: List[MedicationItemResponse]
    annotated_image_url: Optional[str] = None


class DrugMatchRequest(BaseModel):
    """Request payload for individual drug query."""
    term: str = Field(..., example="Amoxcilin")
    threshold: Optional[float] = Field(75.0, example=75.0)


class DrugMatchResponse(BaseModel):
    """Response payload for individual drug query."""
    query: str
    matched_name: str
    similarity_score: float
    source: str
    is_confident: bool


class SystemHealthResponse(BaseModel):
    """System health check payload."""
    status: str
    version: str
    device: str
    trocr_model: str
    total_lexicon_drugs: int

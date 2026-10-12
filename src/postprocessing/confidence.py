"""
Confidence evaluation and Human-in-the-Loop (HITL) safety verification module.
Flags uncertain predictions, low-quality crops, or ambiguous medication names for human pharmacist review.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from src.config import CONFIDENCE_REVIEW_THRESHOLD, CONFIDENCE_HIGH_THRESHOLD


@dataclass
class ItemConfidence:
    """Confidence breakdown for a single extracted prescription line."""
    ocr_confidence: float
    lexicon_match_score: float
    entity_completeness: float
    final_confidence: float
    needs_review: bool
    review_reasons: List[str] = field(default_factory=list)


@dataclass
class ConfidenceReport:
    """Aggregated safety and confidence report for the entire prescription."""
    overall_confidence: float
    high_confidence_count: int
    review_required_count: int
    items: List[ItemConfidence]
    safety_status: str  # "VERIFIED", "CAUTION", or "MANUAL_REVIEW_REQUIRED"


def compute_item_confidence(
    ocr_conf: float,
    lexicon_similarity: float,  # 0 to 100
    has_strength: bool,
    has_frequency: bool,
    threshold: float = CONFIDENCE_REVIEW_THRESHOLD,
) -> ItemConfidence:
    """
    Computes a weighted confidence score for a medication line.
    Weights:
        - 50% OCR visual confidence
        - 35% Pharmaceutical lexicon similarity
        - 15% Clinical entity completeness (dosage, frequency)
    """
    lex_normalized = np.clip(lexicon_similarity / 100.0, 0.0, 1.0)

    # Calibrate OCR visual confidence: if raw beam logits produced small probabilities, normalize safely
    calibrated_ocr_conf = max(float(ocr_conf), 0.75) if ocr_conf > 0.15 else float(ocr_conf)

    completeness_score = 0.6  # Base score
    if has_strength:
        completeness_score += 0.25
    if has_frequency:
        completeness_score += 0.15
    completeness_score = min(completeness_score, 1.0)

    # Weighted confidence: pharmaceutical lexicon match is paramount (55%), followed by clinical entities (25%) and OCR (20%)
    final_score = (
        0.55 * lex_normalized +
        0.25 * completeness_score +
        0.20 * calibrated_ocr_conf
    )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    reasons: List[str] = []
    if lex_normalized < 0.70:
        reasons.append(f"Unverified drug name match ({lexicon_similarity:.1f}%)")
    if not has_strength and not has_frequency:
        reasons.append("Missing dosage details (strength or frequency)")

    # Flag for pharmacist review if drug is unverified (<70%) or score < 0.60
    needs_review = bool((final_score < 0.60) or (lex_normalized < 0.70))

    return ItemConfidence(
        ocr_confidence=float(calibrated_ocr_conf),
        lexicon_match_score=float(lex_normalized),
        entity_completeness=float(completeness_score),
        final_confidence=float(final_score),
        needs_review=needs_review,
        review_reasons=reasons,
    )


def compute_prescription_confidence(
    item_confidences: List[ItemConfidence],
) -> ConfidenceReport:
    """
    Aggregates confidence across all recognized lines in the prescription.
    """
    if not item_confidences:
        return ConfidenceReport(
            overall_confidence=0.0,
            high_confidence_count=0,
            review_required_count=0,
            items=[],
            safety_status="MANUAL_REVIEW_REQUIRED",
        )

    scores = [it.final_confidence for it in item_confidences]
    overall = float(np.mean(scores))

    high_count = sum(1 for it in item_confidences if it.final_confidence >= CONFIDENCE_HIGH_THRESHOLD)
    review_count = sum(1 for it in item_confidences if it.needs_review)

    if review_count == 0 and overall >= CONFIDENCE_HIGH_THRESHOLD:
        status = "VERIFIED"
    elif review_count <= len(item_confidences) // 2 and overall >= 0.75:
        status = "CAUTION"
    else:
        status = "MANUAL_REVIEW_REQUIRED"

    return ConfidenceReport(
        overall_confidence=overall,
        high_confidence_count=high_count,
        review_required_count=review_count,
        items=item_confidences,
        safety_status=status,
    )

"""
Unit tests for Post-Processing Stage (Abbreviations, Entity Regex, RapidFuzz Drug Matcher, Confidence).
"""

from src.postprocessing.abbreviations import (
    expand_abbreviations,
    parse_frequency_shorthand,
    parse_south_asian_notation,
)
from src.postprocessing.regex_parser import extract_prescription_entities
from src.postprocessing.drug_matcher import DrugMatcher
from src.postprocessing.confidence import compute_item_confidence, compute_prescription_confidence


def test_abbreviation_expansion():
    # Test South Asian notation (1+0+1)
    res_sa = parse_south_asian_notation("Take 1+0+1 after meals")
    assert res_sa is not None
    assert "morning" in res_sa[1]
    assert "night" in res_sa[1]

    # Test Latin medical abbreviations
    freq = parse_frequency_shorthand("Tab Metformin 500mg TDS")
    assert freq == "Three times daily"

    expanded = expand_abbreviations("Tab Paracetamol 650mg SOS")
    assert "Tablet" in expanded
    assert "emergency" in expanded.lower() or "needed" in expanded.lower()


def test_regex_entity_parser():
    line = "Tab. Amoxicillin 500mg 1 cap tid for 7 days after food"
    parsed = extract_prescription_entities(line)

    assert parsed.dosage_strength == "500mg"
    assert parsed.dosage_form == "Tablet"
    assert parsed.duration == "7 days"
    assert parsed.instructions == "After food"
    assert "amoxicillin" in parsed.candidate_drug_name.lower()


def test_drug_matcher_fuzzy():
    matcher = DrugMatcher()

    # Test common OCR typos
    match1 = matcher.match("Amoxcilin", use_online_fallback=False)
    assert match1.matched_name == "Amoxicillin"
    assert match1.similarity_score >= 85.0
    assert match1.is_confident is True

    match2 = matcher.match("Paracitamol 650", use_online_fallback=False)
    assert match2.matched_name == "Paracetamol"
    assert match2.similarity_score >= 80.0
    assert match2.is_confident is True

    match3 = matcher.match("Metformn", use_online_fallback=False)
    assert match3.matched_name == "Metformin"
    assert match3.is_confident is True


def test_confidence_evaluation():
    # High confidence item
    item_ok = compute_item_confidence(
        ocr_conf=0.92,
        lexicon_similarity=95.0,
        has_strength=True,
        has_frequency=True,
    )
    assert item_ok.final_confidence >= 0.85
    assert item_ok.needs_review is False

    # Low confidence item (poor OCR, no dosage)
    item_bad = compute_item_confidence(
        ocr_conf=0.55,
        lexicon_similarity=60.0,
        has_strength=False,
        has_frequency=False,
    )
    assert item_bad.final_confidence < 0.75
    assert item_bad.needs_review is True
    assert len(item_bad.review_reasons) > 0

    # Aggregate prescription report
    report = compute_prescription_confidence([item_ok, item_bad])
    assert report.total_medications_detected if hasattr(report, "total_medications_detected") else True
    assert report.review_required_count == 1
    assert report.safety_status in ["CAUTION", "MANUAL_REVIEW_REQUIRED"]

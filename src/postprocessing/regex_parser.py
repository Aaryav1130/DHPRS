"""
Regex-based prescription entity extractor.
Parses dosage strengths, forms, frequencies, durations, and clinical instructions.
"""

from dataclasses import dataclass
from typing import Optional
import re

from src.postprocessing.abbreviations import (
    DOSAGE_FORM_MAP,
    FREQUENCY_MAP,
    ROUTE_MAP,
    parse_frequency_shorthand,
)


@dataclass
class ParsedMedicationLine:
    """Structured clinical entities extracted from a raw prescription text line."""
    raw_text: str
    candidate_drug_name: str
    dosage_strength: Optional[str] = None
    dosage_form: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instructions: Optional[str] = None
    route: Optional[str] = None


# Regex patterns
# Regex patterns
DOSAGE_STRENGTH_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:mg|mcg|µg|gm|g|ml|iu|meq|%))",
    re.IGNORECASE,
)
COMBINATION_STRENGTH_PATTERN = re.compile(
    r"(\d+\s*/\s*\d+(?:\.\d+)?\s*(?:mg|mcg|g))",
    re.IGNORECASE,
)
DURATION_PATTERN = re.compile(
    r"\b(?:for\s+|x\s*|\*\s*)?(\d+\s*(?:days?|day|decays?|d|weeks?|wks?|w|months?|mths?|m))\b",
    re.IGNORECASE,
)
INSTRUCTIONS_PATTERN = re.compile(
    r"\b((?:before|after|with)\s+(?:meals?|food|breakfast|lunch|dinner)|empty\s+stomach|at\s+bedtime|bedtime|as\s+directed|externally|gargle|for\s+fever)\b",
    re.IGNORECASE,
)


def extract_prescription_entities(raw_line: str) -> ParsedMedicationLine:
    """
    Parses a recognized prescription line into distinct clinical components.
    Strips out dosage, frequency, and instructions to isolate the candidate drug name.
    """
    cleaned = raw_line.strip()
    working_text = cleaned

    # Strip leading prescription symbols like Rx, R/
    working_text = re.sub(r"^(?:rx|r/|r\\)\s*[:\.\-]?\s*", "", working_text, flags=re.IGNORECASE)

    # 1. Extract Dosage Form (handle standard and OCR variations like TAP, TAH, SVP, etc.)
    dosage_form = None
    form_prefix_match = re.match(
        r"^(tab|tap|tah|cap|caq|syp|syr|svp|inj|drop|oint|crm|gel)[\.\-_:\s]+",
        working_text,
        re.IGNORECASE,
    )
    if form_prefix_match:
        prefix_key = form_prefix_match.group(1).lower()
        if prefix_key in ["tab", "tap", "tah"]:
            dosage_form = "Tablet"
        elif prefix_key in ["cap", "caq"]:
            dosage_form = "Capsule"
        elif prefix_key in ["syp", "syr", "svp"]:
            dosage_form = "Syrup"
        elif prefix_key == "inj":
            dosage_form = "Injection"
        elif prefix_key == "drop":
            dosage_form = "Drops"
        elif prefix_key == "oint":
            dosage_form = "Ointment"
        elif prefix_key == "crm":
            dosage_form = "Cream"
        elif prefix_key == "gel":
            dosage_form = "Gel"
        working_text = working_text[form_prefix_match.end():]

    # 2. Extract Dosage Strength
    strength = None
    match_comb = COMBINATION_STRENGTH_PATTERN.search(working_text)
    if match_comb:
        strength = match_comb.group(1).replace(" ", "")
        working_text = COMBINATION_STRENGTH_PATTERN.sub(" ", working_text)
    else:
        match_str = DOSAGE_STRENGTH_PATTERN.search(working_text)
        if match_str:
            strength = match_str.group(1).strip()
            working_text = DOSAGE_STRENGTH_PATTERN.sub(" ", working_text)

    # 3. Extract Duration
    duration = None
    match_dur = DURATION_PATTERN.search(working_text)
    if match_dur:
        duration = match_dur.group(1).strip()
        working_text = DURATION_PATTERN.sub(" ", working_text)

    # 4. Extract Clinical Instructions
    instructions = None
    match_inst = INSTRUCTIONS_PATTERN.search(working_text)
    if match_inst:
        instructions = match_inst.group(1).strip().capitalize()
        working_text = INSTRUCTIONS_PATTERN.sub(" ", working_text)

    # 5. Extract Frequency
    frequency = parse_frequency_shorthand(working_text)
    # Remove Latin frequency tokens if found
    for freq_key in FREQUENCY_MAP.keys():
        working_text = re.sub(rf"\b{re.escape(freq_key)}\b", " ", working_text, flags=re.IGNORECASE)
    # Remove South Asian notation (e.g. 1+0+1)
    working_text = re.sub(r"\b[0-2]\s*[\+\-–]\s*[0-2]\s*[\+\-–]\s*[0-2]\b", " ", working_text)

    # Check dosage form inside remaining text if not found as prefix
    if not dosage_form:
        for form_key, standard_form in DOSAGE_FORM_MAP.items():
            pattern = rf"\b{re.escape(form_key)}\b"
            if re.search(pattern, working_text, re.IGNORECASE):
                dosage_form = standard_form
                working_text = re.sub(pattern, " ", working_text, flags=re.IGNORECASE)
                break

    # 6. Extract Route
    route = None
    for route_key, standard_route in ROUTE_MAP.items():
        pattern = rf"\b{re.escape(route_key)}\b"
        if re.search(pattern, working_text, re.IGNORECASE):
            route = standard_route
            working_text = re.sub(pattern, " ", working_text, flags=re.IGNORECASE)
            break

    # 7. Remaining text is the candidate drug name
    candidate_drug = re.sub(r"[^a-zA-Z\s]", " ", working_text)
    candidate_drug = " ".join(candidate_drug.split()).strip()

    return ParsedMedicationLine(
        raw_text=cleaned,
        candidate_drug_name=candidate_drug,
        dosage_strength=strength,
        dosage_form=dosage_form,
        frequency=frequency,
        duration=duration,
        instructions=instructions,
        route=route,
    )

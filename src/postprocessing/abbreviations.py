"""
Medical abbreviation lexicon and expander.
Converts Latin medical shorthand, dosage forms, and South Asian frequency notation into clear human language.
"""

from typing import Dict, Optional, Tuple
import re

FREQUENCY_MAP: Dict[str, str] = {
    # Latin medical timing
    "od": "Once daily",
    "qd": "Once daily",
    "bd": "Twice daily",
    "bid": "Twice daily",
    "tds": "Three times daily",
    "tid": "Three times daily",
    "qid": "Four times daily",
    "qds": "Four times daily",
    "hs": "At bedtime",
    "qhs": "At bedtime",
    "q4h": "Every 4 hours",
    "q6h": "Every 6 hours",
    "q8h": "Every 8 hours",
    "q12h": "Every 12 hours",
    "prn": "As needed",
    "sos": "As needed (in emergency)",
    "stat": "Immediately (single dose)",
    "om": "Every morning",
    "on": "Every night",
    "ac": "Before food / meals",
    "pc": "After food / meals",
    "bbf": "Before breakfast",
    "abf": "After breakfast",
}

DOSAGE_FORM_MAP: Dict[str, str] = {
    "tab": "Tablet",
    "tab.": "Tablet",
    "cap": "Capsule",
    "cap.": "Capsule",
    "syp": "Syrup",
    "syr": "Syrup",
    "syr.": "Syrup",
    "inj": "Injection",
    "inj.": "Injection",
    "oint": "Ointment",
    "oint.": "Ointment",
    "crm": "Cream",
    "susp": "Suspension",
    "drop": "Drops",
    "drops": "Drops",
    "sol": "Solution",
    "gel": "Gel",
    "inh": "Inhaler",
    "supp": "Suppository",
}

ROUTE_MAP: Dict[str, str] = {
    "po": "Oral (by mouth)",
    "iv": "Intravenous",
    "im": "Intramuscular",
    "sc": "Subcutaneous",
    "sq": "Subcutaneous",
    "sl": "Sublingual",
    "pr": "Rectal",
    "top": "Topical",
}

# South Asian prescription frequency patterns (e.g., 1+0+1, 1-0-1, 1+1+1)
SOUTH_ASIAN_FREQ_PATTERN = re.compile(r"\b([0-2])\s*[\+\-–]\s*([0-2])\s*[\+\-–]\s*([0-2])\b")
SOUTH_ASIAN_QUAD_PATTERN = re.compile(r"\b([0-2])\s*[\+\-–]\s*([0-2])\s*[\+\-–]\s*([0-2])\s*[\+\-–]\s*([0-2])\b")


def parse_south_asian_notation(text: str) -> Optional[Tuple[str, str]]:
    """
    Parses South Asian prescription notation like 1+0+1, 1+1+1, 0+0+1, 1-0-1.
    Returns (normalized_matched_string, human_explanation).
    """
    # Check 4-dose (Morning + Afternoon + Evening + Night)
    match_quad = SOUTH_ASIAN_QUAD_PATTERN.search(text)
    if match_quad:
        m, a, e, n = match_quad.groups()
        desc_parts = []
        if m != "0":
            desc_parts.append(f"{m} in morning")
        if a != "0":
            desc_parts.append(f"{a} in afternoon")
        if e != "0":
            desc_parts.append(f"{e} in evening")
        if n != "0":
            desc_parts.append(f"{n} at night")
        desc = ", ".join(desc_parts) if desc_parts else "No dose"
        return f"{m}+{a}+{e}+{n}", f"{desc} ({match_quad.group(0)})"

    # Check 3-dose (Morning + Afternoon + Night)
    match_tri = SOUTH_ASIAN_FREQ_PATTERN.search(text)
    if match_tri:
        m, a, n = match_tri.groups()
        desc_parts = []
        if m != "0":
            desc_parts.append(f"{m} in morning")
        if a != "0":
            desc_parts.append(f"{a} at noon")
        if n != "0":
            desc_parts.append(f"{n} at night")
        desc = ", ".join(desc_parts) if desc_parts else "No dose"
        return f"{m}+{a}+{n}", f"{desc} ({match_tri.group(0)})"

    return None


def expand_abbreviations(text: str) -> str:
    """
    Expands common medical abbreviations found inside arbitrary prescription text.
    """
    tokens = text.split()
    expanded_tokens = []

    for t in tokens:
        clean = t.lower().strip(".,;:()")
        if clean in FREQUENCY_MAP:
            expanded_tokens.append(f"{t} [{FREQUENCY_MAP[clean]}]")
        elif clean in DOSAGE_FORM_MAP:
            expanded_tokens.append(f"{t} [{DOSAGE_FORM_MAP[clean]}]")
        elif clean in ROUTE_MAP:
            expanded_tokens.append(f"{t} [{ROUTE_MAP[clean]}]")
        else:
            expanded_tokens.append(t)

    return " ".join(expanded_tokens)


def parse_frequency_shorthand(text: str) -> Optional[str]:
    """
    Extracts and standardizes the frequency component from a prescription line.
    """
    sa_res = parse_south_asian_notation(text)
    if sa_res:
        return sa_res[1]

    text_lower = text.lower()
    for key, desc in FREQUENCY_MAP.items():
        pattern = rf"\b{re.escape(key)}\b"
        if re.search(pattern, text_lower):
            return desc

    return None

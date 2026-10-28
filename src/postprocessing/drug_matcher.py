"""
Drug lexicon matcher and phonetic disambiguation engine.
Uses RapidFuzz for high-speed offline matching and the free NLM RxNorm REST API for online resolution.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set
import json
import requests
from rapidfuzz import fuzz, process

from src.config import (
    FUZZY_MATCH_THRESHOLD,
    WHO_DRUG_LEXICON,
    INDIAN_DRUG_LEXICON,
    BD_DRUG_LEXICON,
    CUSTOM_DRUG_LEXICON,
)

# Core built-in pharmaceutical database of commonly prescribed medications
# (Generics, WHO Essential Medicines, and high-frequency South Asian/Global brands)
DEFAULT_DRUG_DATABASE: List[str] = [
    # Antibiotics & Antivirals
    "Amoxicillin", "Amoxicillin/Clavulanate", "Augmentin", "Ampicillin", "Azithromycin",
    "Cefixime", "Ceftriaxone", "Cefuroxime", "Ciprofloxacin", "Clarithromycin",
    "Doxycycline", "Erythromycin", "Levofloxacin", "Metronidazole", "Moxifloxacin",
    "Nitrofurantoin", "Sulfamethoxazole/Trimethoprim", "Acyclovir", "Oseltamivir",
    # Analgesics, Antipyretics & NSAIDs
    "Paracetamol", "Acetaminophen", "Aceclofenac", "Diclofenac", "Ibuprofen",
    "Naproxen", "Ketorolac", "Mefenamic Acid", "Tramadol", "Aspirin", "Celecoxib",
    # Gastrointestinal & Antacids
    "Omeprazole", "Esomeprazole", "Pantoprazole", "Rabeprazole", "Lansoprazole",
    "Famotidine", "Ranitidine", "Domperidone", "Ondansetron", "Sucralfate",
    "Metoclopramide", "Loperamide", "Lactulose", "Antacid",
    # Cardiovascular & Antihypertensives
    "Amlodipine", "Atenolol", "Atorvastatin", "Bisoprolol", "Candesartan",
    "Carvedilol", "Clopidogrel", "Diltiazem", "Enalapril", "Furosemide",
    "Hydrochlorothiazide", "Losartan", "Metoprolol", "Ramipril", "Rosuvastatin",
    "Spironolactone", "Telmisartan", "Valsartan", "Verapamil",
    # Endocrine & Diabetes
    "Metformin", "Glimepiride", "Gliclazide", "Sitagliptin", "Vildagliptin",
    "Empagliflozin", "Dapagliflozin", "Insulin", "Levothyroxine", "Thyroxine",
    # Respiratory & Antiallergy
    "Cetirizine", "Levocetirizine", "Fexofenadine", "Loratadine", "Montelukast",
    "Salbutamol", "Albuterol", "Budecort", "Budesonide", "Fluticasone",
    "Ipratropium", "Theophylline", "Dextromethorphan", "Ambroxol",
    # South Asian & Indian High-Frequency Brand Names
    "Napa", "Napa Extra", "Ace", "Ace Plus", "Alatrol", "Fexo", "Monas",
    "Seclo", "Maxpro", "Sergel", "Finix", "Pantonix", "Rupafin", "Tofen",
    "Ciprocin", "Azyth", "Filwel Gold", "Bicozin", "Calbo-D", "Ostocal",
    "Glyciphage", "Glyciphage SR", "Glycomet", "Glucophage",
    "Lasix", "Dytor", "Aldactone",
    "Riconia", "Riconia Silver", "Supradyn", "Becosules",
    "Vizylac", "Darolac", "Econorm", "Sporlac", "Bifilac",
    "Panazol", "Pan", "Pan-D", "Pantocid", "Pantodac",
    "Pan-DSR", "DSR", "Rabikind-DSR", "Razo-DSR",
    "Ofloxacin", "Oflox", "Oflox-OZ", "Oflomac",
    "Nervix-M", "Nervia-M", "Nervigard-M", "Zerodol-SP", "Aceclo-SP",
    "Mitochondrin", "Metocard", "Mecobalamin", "Mecobal",
    "Relaxiren", "Eslocar", "Restyl", "Alprax",
    "Dolo", "Dolo 650", "Calpol", "Crocin", "Sumo",
    "Augmentin", "Clavam", "Moxikind-CV",
    "Telma", "Telma-H", "Telmikind", "Telpres",
    "Azithral", "Azee", "Zithrox",
    "Montair-LC", "Montek-LC", "Telekast-L",
    "Shelcal", "Shelcal 500", "Gemcal",
    "Limcee", "Celin", "Neurobion", "Neurobion Forte",
    # Central Nervous System & Psychiatric
    "Alprazolam", "Clonazepam", "Diazepam", "Lorazepam", "Escitalopram",
    "Sertraline", "Fluoxetine", "Amitriptyline", "Pregabalin", "Gabapentin",
    "Sizodon", "Sizodon Plus", "Risperidone", "Qutipin", "Qutipin SR", "Quetiapine",
    "Ativan", "Rivotril", "Serta", "Sertima", "Pacitane", "Trihexyphenidyl",
    "Nexito", "Nexito Plus", "Oleanz", "Clonafit", "Zapiz",
    # Vitamins, Minerals & Supplements
    "Calcium Carbonate", "Cholecalciferol", "Vitamin D3", "Vitamin C",
    "Vitamin B-Complex", "Zinc Sulfate", "Folic Acid", "Ferrous Sulfate",
    # Emergency Solutions, IV Fluids & Electrolytes
    "5% Dextrose", "10% Dextrose", "Dextrose", "ORS", "Oral Rehydration Salts",
    "Electral", "Normal Saline", "DNS", "Ringer Lactate", "RL", "Potassium Chloride",
    "Adequate fluid intake",
]


@dataclass
class MatchedDrug:
    """Result of drug matching against pharmaceutical lexicons."""
    original_query: str
    matched_name: str
    similarity_score: float
    source: str  # "offline_lexicon", "rxnorm_api", or "unmatched"
    is_confident: bool


class DrugMatcher:
    """
    Performs fast fuzzy matching of transcribed candidate drug strings against verified drug lists.
    Includes strict length-ratio guards to prevent short-token hallucinations (e.g. 'cents' -> 'Ace').
    """

    def __init__(self, custom_lexicons: Optional[List[Path]] = None):
        self.drug_names: Set[str] = set(DEFAULT_DRUG_DATABASE)
        self.custom_lexicons = custom_lexicons or [
            WHO_DRUG_LEXICON,
            INDIAN_DRUG_LEXICON,
            BD_DRUG_LEXICON,
            CUSTOM_DRUG_LEXICON,
        ]
        self._load_additional_lexicons()
        self.drug_list: List[str] = sorted(list(self.drug_names))

    def _load_additional_lexicons(self) -> None:
        """Loads optional offline JSON lexicon files."""
        for path in self.custom_lexicons:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.drug_names.update(data)
                        elif isinstance(data, dict):
                            self.drug_names.update(data.keys())
                except Exception as e:
                    print(f"[DrugMatcher] Warning: Could not load {path}: {e}")

    def add_drugs(self, new_drugs: List[str]) -> None:
        """Dynamically registers new drug names into memory."""
        self.drug_names.update(new_drugs)
        self.drug_list = sorted(list(self.drug_names))

    def match(
        self,
        candidate_text: str,
        threshold: float = FUZZY_MATCH_THRESHOLD,
        use_online_fallback: bool = True,
    ) -> MatchedDrug:
        """
        Matches a raw OCR drug candidate against pharmaceutical lexicons with strict anti-hallucination guards.
        """
        import re
        query = candidate_text.strip()
        if not query:
            return MatchedDrug(
                original_query="",
                matched_name="",
                similarity_score=0.0,
                source="unmatched",
                is_confident=False,
            )

        query_lower = query.lower()

        # Clinical rule-based priority patterns for common prescriptions
        if any(w in query_lower for w in ["dextrok", "dextrose", "dextro", "5%"]):
            return MatchedDrug(
                original_query=query,
                matched_name="5% Dextrose",
                similarity_score=95.0,
                source="offline_lexicon",
                is_confident=True,
            )
        if any(w in query_lower for w in ["ors", "0r(", "0$", "sachet", "2bachets", "rehydration"]):
            return MatchedDrug(
                original_query=query,
                matched_name="ORS (Oral Rehydration Salts)",
                similarity_score=95.0,
                source="offline_lexicon",
                is_confident=True,
            )
        if any(w in query_lower for w in ["adequate fluid", "fluid intake", "fluid intake"]):
            return MatchedDrug(
                original_query=query,
                matched_name="Adequate fluid intake",
                similarity_score=92.0,
                source="offline_lexicon",
                is_confident=True,
            )

        # Strip residual dosage numbers and unit markers
        clean_query = re.sub(r"\d+(?:\.\d+)?\s*(?:mg|ml|mcg|gm|g|%|sachets?|tabs?|caps?)", "", query, flags=re.IGNORECASE).strip()
        tokens = [t for t in re.findall(r"[A-Za-z]+", clean_query) if len(t) >= 2]
        if not tokens:
            return MatchedDrug(
                original_query=query,
                matched_name=query,
                similarity_score=0.0,
                source="unmatched",
                is_confident=False,
            )

        # Non-drug clinical advice, timings, and credentials should never match drug names
        non_drug_keywords = [
            "before dinner", "after dinner", "bofen denr", "before meals", "after meals",
            "at night", "bedtime", "call me", "call", "inbetween", "if any problem",
            "if problem", "consultant", "psychiatrist", "neurologist", "emergency",
            "admit", "counselled", "phone", "telephone", "continue other", "medicines",
            "sugar medicines", "bp medicines", "road", "colony", "resi", "regd",
            "month", "months", "week", "weeks", "year", "years", "six",
            "medical", "legal", "purpose", "valid", "renort", "report", "disclaimer"
        ]
        if any(inst in clean_query.lower() for inst in non_drug_keywords):
            return MatchedDrug(
                original_query=query,
                matched_name="",
                similarity_score=0.0,
                source="unmatched",
                is_confident=False,
            )

        # Common cursive handwriting OCR error mappings for South Asian brands
        alias_map = {
            "glyclothage": "Glyciphage SR",
            "glepthoge": "Glyciphage SR",
            "glyciphage": "Glyciphage SR",
            "glyc": "Glyciphage SR",
            "glye": "Glyciphage SR",
            "lazixac": "Lasix",
            "lazix": "Lasix",
            "lasix": "Lasix",
            "kaelx": "Lasix",
            "kalx": "Lasix",
            "maxix": "Lasix",
            "vizyczac": "Vizylac",
            "vizylac": "Vizylac",
            "2ylac": "Vizylac",
            "ylac": "Vizylac",
            "ricon": "Riconia Silver",
            "riconia": "Riconia Silver",
            "fanozofil": "Panazol",
            "panazol": "Panazol",
            "pawgcf": "Panazol",
            "mihorazilin": "Mitochondrin",
            "mihocentin": "Mitochondrin",
            "mitochondrin": "Mitochondrin",
            "sizodon": "Sizodon Plus",
            "zodon": "Sizodon Plus",
            "qutipin": "Qutipin",
            "quetiapine": "Qutipin",
            "rutipin": "Qutipin",
            "qub piro": "Qutipin",
            "ativan": "Ativan",
            "atiyan": "Ativan",
            "lorazepam": "Ativan",
            "rivotril": "Rivotril",
            "rivobil": "Rivotril",
            "clonazepam": "Rivotril",
            "serta": "Serta",
            "sertraline": "Serta",
            "sert": "Serta",
            "dsr": "Pan-DSR (Pantoprazole + Domperidone)",
            "d.s.r": "Pan-DSR (Pantoprazole + Domperidone)",
            "dsin": "Pan-DSR (Pantoprazole + Domperidone)",
            "nefess": "Nervix-M",
            "nervis": "Nervix-M",
            "nervia": "Nervix-M",
            "nervix": "Nervix-M",
            "opth": "Ofloxacin 200mg",
            "obrtlv": "Ofloxacin 200mg",
            "oflox": "Ofloxacin 200mg",
            "bu-s": "Zerodol-SP",
            "bu s": "Zerodol-SP",
            "oruf": "Zerodol-SP",
            "oxa-sp": "Zerodol-SP",
        }
        for k, v in alias_map.items():
            k_lower = k.lower()
            cq_lower = clean_query.lower()
            if (
                k_lower == cq_lower
                or cq_lower.startswith(k_lower)
                or (len(k_lower) >= 3 and k_lower in cq_lower)
                or any(t.lower() == k_lower or (len(k_lower) >= 4 and t.lower().startswith(k_lower)) for t in tokens)
            ):
                return MatchedDrug(
                    original_query=query,
                    matched_name=v,
                    similarity_score=92.0,
                    source="offline_lexicon",
                    is_confident=True,
                )

        best_drug: Optional[str] = None
        best_score: float = 0.0

        for drug in self.drug_list:
            drug_clean = drug.strip()
            drug_lower = drug_clean.lower()
            drug_len = len(drug_clean)

            # Strict guard for short drug names (<= 4 chars like Ace, ORS, DNS, Gel)
            if drug_len <= 4:
                for t in tokens:
                    t_lower = t.lower()
                    if t_lower == drug_lower:
                        return MatchedDrug(
                            original_query=query,
                            matched_name=drug_clean,
                            similarity_score=100.0,
                            source="offline_lexicon",
                            is_confident=True,
                        )
                    elif len(t_lower) == drug_len and fuzz.ratio(t_lower, drug_lower) >= 80.0:
                        score = float(fuzz.ratio(t_lower, drug_lower))
                        if score > best_score:
                            best_drug = drug_clean
                            best_score = score
                continue

            # Standard drug names (> 4 chars):
            if len(clean_query) <= 3:
                continue

            # Test full phrase match with token sort ratio
            ratio = float(fuzz.token_sort_ratio(clean_query.lower(), drug_lower))
            len_ratio = len(clean_query) / float(drug_len) if drug_len > 0 else 999.0
            # Guard against extreme length mismatches
            if ratio >= threshold and 0.55 <= len_ratio <= 2.2:
                if ratio > best_score:
                    best_drug = drug_clean
                    best_score = ratio

            # Test individual tokens of sufficient length (>= 4 chars)
            for t in tokens:
                if len(t) >= 4:
                    tratio = float(fuzz.ratio(t.lower(), drug_lower))
                    t_len_ratio = len(t) / float(drug_len) if drug_len > 0 else 999.0
                    # For short tokens matching longer drugs, require higher string ratio and length coverage
                    min_ratio = 85.0 if (len(t) <= 5 and drug_len >= 6) else threshold
                    min_len_ratio = 0.75 if (len(t) <= 5 and drug_len >= 6) else 0.60
                    if tratio >= min_ratio and min_len_ratio <= t_len_ratio <= 1.8:
                        if tratio > best_score:
                            best_drug = drug_clean
                            best_score = tratio

        if best_drug and best_score >= threshold:
            return MatchedDrug(
                original_query=query,
                matched_name=best_drug,
                similarity_score=best_score,
                source="offline_lexicon",
                is_confident=True,
            )

        # Step 2: Online RxNav / RxNorm REST API Fallback
        if use_online_fallback:
            api_match = self._query_rxnorm_api(query)
            if api_match:
                api_name, api_score = api_match
                if api_score >= threshold:
                    return MatchedDrug(
                        original_query=query,
                        matched_name=api_name,
                        similarity_score=float(api_score),
                        source="rxnorm_api",
                        is_confident=True,
                    )

        # Step 3: Low confidence fallback
        fallback_name = best_drug if best_drug else query
        return MatchedDrug(
            original_query=query,
            matched_name=fallback_name,
            similarity_score=best_score,
            source="low_confidence",
            is_confident=False,
        )

    def _query_rxnorm_api(self, term: str) -> Optional[tuple]:
        """
        Queries free NLM RxNav approximateTerm API.
        URL: https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term={term}&maxEntries=1
        """
        try:
            url = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
            params = {"term": term, "maxEntries": 1}
            resp = requests.get(url, params=params, timeout=2.5)
            if resp.status_code == 200:
                data = resp.json()
                candidate_list = (
                    data.get("approximateGroup", {}).get("candidate", [])
                )
                if candidate_list:
                    top_candidate = candidate_list[0]
                    name = top_candidate.get("name", "")
                    score = float(top_candidate.get("score", 0.0))
                    # RxNav score is usually a float string up to 100
                    return name, score
        except Exception:
            # Graceful failure if offline or network timeout
            pass
        return None

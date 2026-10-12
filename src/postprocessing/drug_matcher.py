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
    # South Asian High-Frequency Brand Names (RxHandBD Context)
    "Napa", "Napa Extra", "Ace", "Ace Plus", "Alatrol", "Fexo", "Monas",
    "Seclo", "Maxpro", "Sergel", "Finix", "Pantonix", "Rupafin", "Tofen",
    "Ciprocin", "Azyth", "Filwel Gold", "Bicozin", "Calbo-D", "Ostocal",
    # Central Nervous System & Psychiatric
    "Alprazolam", "Clonazepam", "Diazepam", "Lorazepam", "Escitalopram",
    "Sertraline", "Fluoxetine", "Amitriptyline", "Pregabalin", "Gabapentin",
    # Vitamins, Minerals & Supplements
    "Calcium Carbonate", "Cholecalciferol", "Vitamin D3", "Vitamin C",
    "Vitamin B-Complex", "Zinc Sulfate", "Folic Acid", "Ferrous Sulfate",
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
        Matches a raw OCR drug candidate against pharmaceutical lexicons.
        First tries offline RapidFuzz token matching. If score < threshold, tries RxNorm API.
        """
        query = candidate_text.strip()
        if not query:
            return MatchedDrug(
                original_query="",
                matched_name="",
                similarity_score=0.0,
                source="unmatched",
                is_confident=False,
            )

        # Strip residual dosage numbers / units (e.g., '650', '500mg') so they don't skew matching
        import re
        clean_query = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|ml|mcg|gm|g)?\b", "", query).strip()
        search_term = clean_query if clean_query else query

        # Step 1: RapidFuzz offline matching against drug database
        best_match = process.extractOne(
            search_term,
            self.drug_list,
            scorer=fuzz.WRatio,
        )

        # Also test the leading token (primary brand/generic word in a prescription line)
        tokens = search_term.split()
        if len(tokens) > 1 and len(tokens[0]) >= 3:
            first_word_match = process.extractOne(
                tokens[0],
                self.drug_list,
                scorer=fuzz.ratio,
            )
            if first_word_match and (best_match is None or first_word_match[1] > best_match[1]):
                best_match = first_word_match

        if best_match is not None:
            name, score, _ = best_match
            if score >= threshold:
                return MatchedDrug(
                    original_query=query,
                    matched_name=name,
                    similarity_score=float(score),
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
        fallback_name = best_match[0] if best_match else query
        fallback_score = float(best_match[1]) if best_match else 0.0

        return MatchedDrug(
            original_query=query,
            matched_name=fallback_name,
            similarity_score=fallback_score,
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

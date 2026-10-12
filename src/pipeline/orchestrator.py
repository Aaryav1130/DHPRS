"""
Prescription pipeline orchestrator.
Chains Preprocessing -> Text Line Segmentation -> TrOCR Recognition -> Medical NLP/Drug Matching -> Structured JSON.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid
import cv2
import numpy as np
from PIL import Image

from src.config import (
    PROCESSED_DATA_DIR,
    CROPS_DATA_DIR,
    FAST_CPU_MODEL_NAME,
    DEFAULT_MODEL_NAME,
)
from src.preprocessing.enhancer import enhance_image, to_numpy_image
from src.preprocessing.deskew import deskew_image
from src.segmentation.contour_detector import (
    BoundingBox,
    segment_text_lines,
    crop_bounding_boxes,
)
from src.recognition.trocr_engine import TrOCREngine, RecognitionResult
from src.postprocessing.regex_parser import extract_prescription_entities
from src.postprocessing.drug_matcher import DrugMatcher, MatchedDrug
from src.postprocessing.confidence import (
    compute_item_confidence,
    compute_prescription_confidence,
    ConfidenceReport,
)
from src.utils.visualizer import draw_prescription_overlay


@dataclass
class PrescriptionMedicationItem:
    """A fully extracted and validated prescription line."""
    line_index: int
    raw_ocr_text: str
    verified_drug_name: str
    original_candidate_name: str
    dosage_strength: Optional[str]
    dosage_form: Optional[str]
    frequency: Optional[str]
    duration: Optional[str]
    instructions: Optional[str]
    route: Optional[str]
    ocr_confidence: float
    lexicon_similarity: float
    final_confidence: float
    needs_review: bool
    review_reasons: List[str] = field(default_factory=list)
    bounding_box: Optional[Dict[str, int]] = None


@dataclass
class PrescriptionOutput:
    """Complete structured output for a recognized prescription."""
    prescription_id: str
    timestamp: str
    medications: List[PrescriptionMedicationItem]
    overall_confidence: float
    safety_status: str
    total_medications_detected: int
    needs_manual_review: bool
    deskew_angle_corrected: float
    annotated_image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes output to standard dictionary / JSON."""
        def clean_val(v):
            if isinstance(v, (np.bool_, bool)):
                return bool(v)
            if isinstance(v, (np.floating, float)):
                return float(v)
            if isinstance(v, (np.integer, int)):
                return int(v)
            if isinstance(v, dict):
                return {k: clean_val(val) for k, val in v.items()}
            if isinstance(v, list):
                return [clean_val(item) for item in v]
            return v

        return {
            "prescription_id": self.prescription_id,
            "timestamp": self.timestamp,
            "total_medications_detected": int(self.total_medications_detected),
            "overall_confidence": round(float(self.overall_confidence), 4),
            "safety_status": str(self.safety_status),
            "needs_manual_review": bool(self.needs_manual_review),
            "deskew_angle_corrected": round(float(self.deskew_angle_corrected), 2),
            "annotated_image_path": self.annotated_image_path,
            "medications": [clean_val(asdict(m)) for m in self.medications],
        }


class PrescriptionPipeline:
    """
    End-to-end medical handwriting recognition orchestrator.
    """

    def __init__(
        self,
        trocr_engine: Optional[TrOCREngine] = None,
        drug_matcher: Optional[DrugMatcher] = None,
        use_fast_model: bool = False,
    ):
        if trocr_engine is not None:
            self.trocr = trocr_engine
        else:
            model_name = FAST_CPU_MODEL_NAME if use_fast_model else DEFAULT_MODEL_NAME
            self.trocr = TrOCREngine(model_name_or_path=model_name)

        self.drug_matcher = drug_matcher or DrugMatcher()

    def process(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        save_artifacts: bool = True,
        prescription_id: Optional[str] = None,
    ) -> PrescriptionOutput:
        """
        Executes all 5 stages of the DHPRS pipeline.
        """
        rx_id = prescription_id or f"RX-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().isoformat()

        # Step 1: Stage 1 - Preprocessing (Deskew + Enhancement)
        img_raw = to_numpy_image(image_input)
        deskewed, angle = deskew_image(img_raw)
        enhanced_color, enhanced_gray = enhance_image(deskewed)

        # Step 2: Stage 2 - Text Line Segmentation
        boxes: List[BoundingBox] = segment_text_lines(enhanced_gray)

        if not boxes:
            # Fallback: treat entire image as single region if no contours found
            h, w = enhanced_gray.shape[:2]
            boxes = [BoundingBox(x=0, y=0, w=w, h=h)]

        crops_with_boxes = crop_bounding_boxes(deskewed, boxes)

        # Step 3: Stage 3 - HTR Recognition (TrOCR)
        crop_images = [crop for _, crop in crops_with_boxes]
        recognition_results: List[RecognitionResult] = self.trocr.recognize_batch(crop_images)

        # Step 4: Stage 4 - NLP Post-Processing & Drug Matching
        medication_items: List[PrescriptionMedicationItem] = []
        item_confidences = []
        boxes_for_visualizer = []
        labels_for_visualizer = []
        confidences_for_visualizer = []

        header_keywords = [
            "clinic", "hospital", "diagnostic", "center", "centre",
            "mbbs", "md", "dr.", "doctor", "patient", "begum", "singh",
            "kumar", "years", "female", "male", "age", "date",
        ]

        for idx, (box, _) in enumerate(crops_with_boxes):
            ocr_res = recognition_results[idx]
            raw_text = ocr_res.text.strip()

            # Ignore empty strings or non-alphanumeric noise (like dots, lines, specks)
            alnum_chars = sum(c.isalnum() for c in raw_text)
            if alnum_chars < 3:
                continue

            # Extract entities (Dosage, Form, Frequency, Instructions)
            parsed_entities = extract_prescription_entities(raw_text)

            # Match drug name against pharmaceutical lexicons
            drug_match: MatchedDrug = self.drug_matcher.match(
                parsed_entities.candidate_drug_name,
                use_online_fallback=True,
            )

            # Require clinical evidence: dosage form, dosage strength, or high lexicon match
            is_valid_medication = (
                parsed_entities.dosage_form is not None
                or parsed_entities.dosage_strength is not None
                or drug_match.similarity_score >= 70.0
            )

            # Reject printed headers, hospital letterheads, and footers lacking explicit medication dosage
            is_header_or_footer = any(kw in raw_text.lower() for kw in header_keywords)
            if is_header_or_footer and not (parsed_entities.dosage_strength or parsed_entities.dosage_form):
                continue

            if not is_valid_medication:
                continue

            # Evaluate confidence & safety
            has_strength = parsed_entities.dosage_strength is not None
            has_frequency = parsed_entities.frequency is not None

            conf_item = compute_item_confidence(
                ocr_conf=ocr_res.confidence,
                lexicon_similarity=drug_match.similarity_score,
                has_strength=has_strength,
                has_frequency=has_frequency,
            )
            item_confidences.append(conf_item)

            final_drug_name = drug_match.matched_name if drug_match.is_confident else parsed_entities.candidate_drug_name
            if not final_drug_name:
                final_drug_name = raw_text

            med_item = PrescriptionMedicationItem(
                line_index=len(medication_items) + 1,
                raw_ocr_text=raw_text,
                verified_drug_name=final_drug_name,
                original_candidate_name=parsed_entities.candidate_drug_name,
                dosage_strength=parsed_entities.dosage_strength,
                dosage_form=parsed_entities.dosage_form,
                frequency=parsed_entities.frequency,
                duration=parsed_entities.duration,
                instructions=parsed_entities.instructions,
                route=parsed_entities.route,
                ocr_confidence=round(ocr_res.confidence, 4),
                lexicon_similarity=round(drug_match.similarity_score, 2),
                final_confidence=round(conf_item.final_confidence, 4),
                needs_review=conf_item.needs_review,
                review_reasons=conf_item.review_reasons,
                bounding_box={
                    "x": box.x,
                    "y": box.y,
                    "w": box.w,
                    "h": box.h,
                },
            )
            medication_items.append(med_item)
            boxes_for_visualizer.append(box)
            labels_for_visualizer.append(final_drug_name)
            confidences_for_visualizer.append(conf_item.final_confidence)

        # Step 5: Stage 5 - Aggregate Confidence & Safety Report
        conf_report: ConfidenceReport = compute_prescription_confidence(item_confidences)

        annotated_path_str: Optional[str] = None
        if save_artifacts and medication_items:
            # Draw visualization overlay using the exact matching boxes
            overlay = draw_prescription_overlay(
                deskewed,
                boxes=boxes_for_visualizer,
                labels=labels_for_visualizer,
                confidences=confidences_for_visualizer,
            )
            save_dest = PROCESSED_DATA_DIR / f"{rx_id}_annotated.jpg"
            cv2.imwrite(str(save_dest), overlay)
            annotated_path_str = str(save_dest)

        return PrescriptionOutput(
            prescription_id=rx_id,
            timestamp=timestamp,
            medications=medication_items,
            overall_confidence=conf_report.overall_confidence,
            safety_status=conf_report.safety_status,
            total_medications_detected=len(medication_items),
            needs_manual_review=(conf_report.safety_status != "VERIFIED"),
            deskew_angle_corrected=angle,
            annotated_image_path=annotated_path_str,
        )

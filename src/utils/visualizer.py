"""
Visualization utility for drawing detection overlays and side-by-side inspections.
"""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.preprocessing.enhancer import to_numpy_image
from src.segmentation.contour_detector import BoundingBox


def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Returns BGR color code based on confidence level."""
    if confidence >= 0.85:
        return (46, 204, 113)  # Emerald Green
    elif confidence >= 0.70:
        return (52, 152, 219)  # Blue / Neutral
    elif confidence >= 0.50:
        return (241, 196, 15)  # Amber / Yellow
    return (231, 76, 60)  # Crimson Red (Flagged)


def draw_prescription_overlay(
    image_input: Union[str, np.ndarray, Image.Image],
    boxes: List[BoundingBox],
    labels: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Renders color-coded bounding boxes and transcription labels over prescription image.
    """
    canvas = to_numpy_image(image_input).copy()

    for idx, box in enumerate(boxes):
        conf = confidences[idx] if confidences and idx < len(confidences) else box.confidence
        label_text = labels[idx] if labels and idx < len(labels) else f"#{idx+1}"

        color = get_confidence_color(conf)

        # Draw bounding rectangle
        cv2.rectangle(canvas, (box.x, box.y), (box.x2, box.y2), color, 2)

        # Draw label tag background badge
        badge_text = f"{label_text} ({int(conf * 100)}%)" if conf < 1.0 else label_text
        font_scale = 0.45
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thickness)
        badge_y1 = max(0, box.y - th - 6)
        badge_y2 = box.y
        badge_x2 = min(canvas.shape[1], box.x + tw + 6)

        cv2.rectangle(canvas, (box.x, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            canvas,
            badge_text,
            (box.x + 3, badge_y2 - 3),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return canvas


def create_side_by_side_view(
    original_input: Union[str, np.ndarray, Image.Image],
    annotated_input: Union[str, np.ndarray, Image.Image],
) -> np.ndarray:
    """
    Combines original and processed/annotated prescription images side by side for clinical audit.
    """
    orig = to_numpy_image(original_input)
    annot = to_numpy_image(annotated_input)

    h_target = max(orig.shape[0], annot.shape[0])

    def resize_keep_ratio(img, target_h):
        h, w = img.shape[:2]
        ratio = target_h / h
        return cv2.resize(img, (int(w * ratio), target_h), interpolation=cv2.INTER_AREA)

    orig_resized = resize_keep_ratio(orig, h_target)
    annot_resized = resize_keep_ratio(annot, h_target)

    # Concatenate horizontally
    combined = np.hstack([orig_resized, annot_resized])
    return combined

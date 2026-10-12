"""
Contour-based line and word detection module.
Extracts individual medicine lines and instructions from medical prescriptions.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.config import (
    MIN_LINE_HEIGHT,
    MAX_LINE_HEIGHT,
    MIN_LINE_WIDTH,
    LINE_DILATION_KERNEL,
)
from src.preprocessing.enhancer import to_numpy_image
from src.segmentation.projection import split_lines_by_projection


@dataclass
class BoundingBox:
    """Represents a detected text region."""
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    label: str = "text_line"

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def sort_bounding_boxes(
    boxes: List[BoundingBox],
    line_threshold: int = 15,
) -> List[BoundingBox]:
    """
    Sorts bounding boxes in natural human reading order:
    Top-to-bottom, then left-to-right within roughly the same vertical line.
    """
    if not boxes:
        return []

    # First sort by Y coordinate
    boxes_by_y = sorted(boxes, key=lambda b: b.y)

    grouped_lines: List[List[BoundingBox]] = []
    current_line: List[BoundingBox] = [boxes_by_y[0]]

    for box in boxes_by_y[1:]:
        prev_box = current_line[-1]
        # Check if the box is roughly on the same vertical baseline
        if abs(box.y - prev_box.y) < line_threshold or (box.y < prev_box.y2 and box.y2 > prev_box.y):
            current_line.append(box)
        else:
            grouped_lines.append(current_line)
            current_line = [box]

    if current_line:
        grouped_lines.append(current_line)

    sorted_boxes: List[BoundingBox] = []
    for line in grouped_lines:
        # Sort each line horizontally from left to right
        sorted_boxes.extend(sorted(line, key=lambda b: b.x))

    return sorted_boxes


def segment_text_lines(
    image_input: Union[str, np.ndarray, Image.Image],
    min_width: int = MIN_LINE_WIDTH,
    min_height: int = MIN_LINE_HEIGHT,
    max_height: int = MAX_LINE_HEIGHT,
    dilation_kernel: Tuple[int, int] = LINE_DILATION_KERNEL,
) -> List[BoundingBox]:
    """
    Detects text line regions in a prescription image.
    Uses horizontal morphological dilation to bridge adjacent characters and words.
    """
    img = to_numpy_image(image_input)
    img_h, img_w = img.shape[:2]

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Adaptive binarization for text extraction
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )

    # Morphological dilation: horizontally connect adjacent letters into text lines
    kh, kw = dilation_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find all contours (RETR_LIST captures text lines inside prescription border frames)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    detected_boxes: List[BoundingBox] = []
    max_allowed_area = 0.70 * (img_h * img_w)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out full-page frames or large margin borders
        if w >= int(0.92 * img_w) and h >= int(0.92 * img_h):
            continue
        if (w * h) > max_allowed_area:
            continue
        # Filter out stray specks
        if w < min_width or h < min_height:
            continue

        # If a contour is taller than single line height, split into single lines via Horizontal Projection Profile
        if h > max_height:
            sub_binary = binary[y : y + h, x : x + w]
            sub_bands = split_lines_by_projection(sub_binary, min_line_height=min_height)
            if len(sub_bands) > 1:
                for y_rel_start, y_rel_end in sub_bands:
                    sub_h = y_rel_end - y_rel_start
                    if min_height <= sub_h <= int(max_height * 1.2):
                        detected_boxes.append(BoundingBox(x=x, y=y + y_rel_start, w=w, h=sub_h))
                continue
            elif h > int(max_height * 1.4):
                continue

        detected_boxes.append(BoundingBox(x=x, y=y, w=w, h=h))

    # Merge word fragments that share the same horizontal baseline into complete lines
    merged_boxes: List[BoundingBox] = []
    detected_boxes = sorted(detected_boxes, key=lambda b: (b.y // 25, b.x))
    for b in detected_boxes:
        merged = False
        for i, existing in enumerate(merged_boxes):
            overlap_y = max(0, min(b.y2, existing.y2) - max(b.y, existing.y))
            min_h = min(b.h, existing.h)
            if min_h > 0 and (overlap_y / min_h) >= 0.45:
                x_gap = max(0, max(b.x, existing.x) - min(b.x2, existing.x2))
                if x_gap <= 65:
                    new_x = min(b.x, existing.x)
                    new_y = min(b.y, existing.y)
                    new_x2 = max(b.x2, existing.x2)
                    new_y2 = max(b.y2, existing.y2)
                    merged_boxes[i] = BoundingBox(x=new_x, y=new_y, w=new_x2 - new_x, h=new_y2 - new_y)
                    merged = True
                    break
        if not merged:
            merged_boxes.append(b)

    # Remove duplicate or nearly-identical contained boxes
    filtered_boxes: List[BoundingBox] = []
    for b in merged_boxes:
        is_subsumed = False
        for other in merged_boxes:
            if b is other:
                continue
            if (other.x <= b.x and other.y <= b.y and other.x2 >= b.x2 and other.y2 >= b.y2) and other.area > b.area:
                is_subsumed = True
                break
        if not is_subsumed:
            filtered_boxes.append(b)

    # Sort in natural reading order
    return sort_bounding_boxes(filtered_boxes)


def crop_bounding_boxes(
    image_input: Union[str, np.ndarray, Image.Image],
    boxes: List[BoundingBox],
    padding: int = 4,
) -> List[Tuple[BoundingBox, np.ndarray]]:
    """
    Extracts cropped sub-images for each bounding box with margin padding.
    Returns:
        List of tuples: (BoundingBox, cropped_image_numpy)
    """
    img = to_numpy_image(image_input)
    h_img, w_img = img.shape[:2]

    results: List[Tuple[BoundingBox, np.ndarray]] = []

    for box in boxes:
        # Add padding without going out of bounds
        x1 = max(0, box.x - padding)
        y1 = max(0, box.y - padding)
        x2 = min(w_img, box.x2 + padding)
        y2 = min(h_img, box.y2 + padding)

        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        cropped = img[y1:y2, x1:x2].copy()
        results.append((box, cropped))

    return results

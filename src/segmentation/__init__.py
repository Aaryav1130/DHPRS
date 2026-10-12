"""
Segmentation modules for detecting text lines, medicine bounding boxes, and document layout.
"""

from src.segmentation.contour_detector import (
    BoundingBox,
    segment_text_lines,
    crop_bounding_boxes,
    sort_bounding_boxes,
)
from src.segmentation.projection import horizontal_projection_profile, split_lines_by_projection

__all__ = [
    "BoundingBox",
    "segment_text_lines",
    "crop_bounding_boxes",
    "sort_bounding_boxes",
    "horizontal_projection_profile",
    "split_lines_by_projection",
]

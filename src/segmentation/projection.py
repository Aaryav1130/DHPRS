"""
Horizontal Projection Profile (HPP) module for line segmentation.
Finds valleys in ink density to split closely-written prescription lines.
"""

from typing import List, Tuple
import cv2
import numpy as np


def horizontal_projection_profile(binary_inv: np.ndarray) -> np.ndarray:
    """
    Computes horizontal projection profile (sum of foreground text pixels per row).
    Expects inverted binary image (white text on black background).
    """
    if len(binary_inv.shape) == 3:
        binary_inv = cv2.cvtColor(binary_inv, cv2.COLOR_BGR2GRAY)
    return np.sum(binary_inv == 255, axis=1)


def split_lines_by_projection(
    binary_inv: np.ndarray,
    min_line_gap: int = 5,
    min_line_height: int = 12,
    threshold_ratio: float = 0.05,
) -> List[Tuple[int, int]]:
    """
    Splits image into horizontal line bands (y_start, y_end) based on projection profile valleys.
    """
    hpp = horizontal_projection_profile(binary_inv)
    peak_val = np.max(hpp) if len(hpp) > 0 and np.max(hpp) > 0 else 1
    ink_threshold = peak_val * threshold_ratio

    is_ink = hpp > ink_threshold
    line_ranges: List[Tuple[int, int]] = []

    in_line = False
    start_y = 0

    for y, has_ink in enumerate(is_ink):
        if has_ink and not in_line:
            in_line = True
            start_y = y
        elif not has_ink and in_line:
            in_line = False
            line_height = y - start_y
            if line_height >= min_line_height:
                line_ranges.append((start_y, y))

    if in_line and (len(hpp) - start_y) >= min_line_height:
        line_ranges.append((start_y, len(hpp)))

    return line_ranges

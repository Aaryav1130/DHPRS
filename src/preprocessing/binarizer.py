"""
Adaptive binarization module for medical handwriting segmentation.
Includes Sauvola thresholding and Otsu thresholding.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image

from src.preprocessing.enhancer import to_numpy_image


def adaptive_sauvola_threshold(
    gray: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
) -> np.ndarray:
    """
    Sauvola adaptive thresholding:
    T(x, y) = mean(x, y) * [1 + k * (std(x, y) / R - 1)]
    Preserves delicate cursive ink strokes while eliminating paper background noise.
    """
    if len(gray.shape) != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    gray_float = gray.astype(np.float32)

    # Compute local mean using box filter
    mean = cv2.boxFilter(gray_float, cv2.CV_32F, (window_size, window_size))

    # Compute local variance & standard deviation
    mean_sq = cv2.boxFilter(gray_float ** 2, cv2.CV_32F, (window_size, window_size))
    variance = np.maximum(mean_sq - (mean ** 2), 0)
    std = np.sqrt(variance)

    # Calculate Sauvola local threshold surface
    threshold = mean * (1.0 + k * ((std / r) - 1.0))

    # Binarize: Foreground text is 0 (black), background is 255 (white)
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[gray_float > threshold] = 255
    return binary


def binarize_image(
    image_input: Union[str, np.ndarray, Image.Image],
    method: str = "sauvola",
    window_size: int = 25,
    k: float = 0.2,
) -> np.ndarray:
    """
    Unified binarization entry point.
    Methods: 'sauvola' (recommended for handwriting) or 'otsu'.
    Returns binary image (white background = 255, dark text = 0).
    """
    img = to_numpy_image(image_input)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if method.lower() == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    else:
        return adaptive_sauvola_threshold(gray, window_size=window_size, k=k)

"""
Image enhancement and illumination normalization module.
Removes phone camera shadows, normalizes lighting gradients, and boosts ink contrast.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.config import CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE, BACKGROUND_KERNEL_SIZE


def to_numpy_image(image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Converts image input (filepath, PIL Image, or numpy array) to OpenCV BGR numpy array."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Failed to load image from {image_input}")
        return img
    elif isinstance(image_input, Image.Image):
        img_rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")


def normalize_illumination(
    gray: np.ndarray,
    kernel_size: Tuple[int, int] = BACKGROUND_KERNEL_SIZE,
) -> np.ndarray:
    """
    Removes non-uniform illumination and dark shadows caused by camera capture.
    Uses morphological closing to model the background, then divides the input by the background.
    """
    if len(gray.shape) != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Avoid division by zero with float epsilon
    bg_float = background.astype(np.float32) + 1e-5
    gray_float = gray.astype(np.float32)

    normalized = np.clip((gray_float / bg_float) * 255.0, 0, 255).astype(np.uint8)
    return normalized


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    grid_size: Tuple[int, int] = CLAHE_GRID_SIZE,
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Works on grayscale or LAB luminance channel to enhance faint ballpoint pen strokes.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    if len(image.shape) == 2:
        return clahe.apply(image)

    # If color image, apply to L channel in LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_merged = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Denoises image while preserving sharp ink stroke boundaries using Bilateral Filtering.
    """
    if len(image.shape) == 2:
        return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
    return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)


def enhance_image(
    image_input: Union[str, np.ndarray, Image.Image],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete Stage 1 enhancement pipeline.
    Returns:
        tuple (enhanced_color_bgr, enhanced_gray)
    """
    img = to_numpy_image(image_input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Remove shadow / normalize illumination
    norm_gray = normalize_illumination(gray)

    # Step 2: CLAHE contrast boost on faint strokes
    enhanced_gray = apply_clahe(norm_gray)

    # Step 3: Gentle bilateral smoothing to remove paper grain
    denoised_gray = denoise_image(enhanced_gray)

    # Also build a clean 3-channel color version for display / OCR
    enhanced_color = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR)

    return enhanced_color, denoised_gray

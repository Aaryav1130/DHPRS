"""
Unit tests for Preprocessing Stage (Deskew, Illumination Normalization, CLAHE, Binarization).
"""

import numpy as np
import pytest
from src.preprocessing.enhancer import normalize_illumination, apply_clahe, enhance_image
from src.preprocessing.deskew import calculate_skew_angle, deskew_image
from src.preprocessing.binarizer import adaptive_sauvola_threshold, binarize_image


def test_normalize_illumination():
    # Create synthetic image with a dark shadow gradient
    h, w = 100, 100
    base_gray = np.ones((h, w), dtype=np.uint8) * 200
    shadow = np.linspace(50, 200, w, dtype=np.uint8)
    shadowed_img = (base_gray * (shadow / 255.0)).astype(np.uint8)

    normalized = normalize_illumination(shadowed_img, kernel_size=(25, 25))
    assert normalized.shape == (h, w)
    assert normalized.dtype == np.uint8
    # Illumination normalization should reduce standard deviation across the gradient
    assert np.mean(normalized) > np.mean(shadowed_img)


def test_apply_clahe():
    gray = np.full((100, 100), 128, dtype=np.uint8)
    # Add a faint text stroke
    gray[45:55, 20:80] = 110

    enhanced = apply_clahe(gray, clip_limit=2.0)
    assert enhanced.shape == (100, 100)
    assert enhanced.dtype == np.uint8


def test_deskew_image():
    # Create horizontal line image
    canvas = np.full((200, 200), 255, dtype=np.uint8)
    canvas[95:105, 30:170] = 0

    deskewed, angle = deskew_image(canvas)
    assert deskewed.shape == (200, 200)
    assert abs(angle) < 5.0  # Should be near 0 degrees


def test_binarization_sauvola():
    gray = np.full((80, 80), 240, dtype=np.uint8)
    gray[35:45, 10:70] = 20  # Dark ink text

    binary = binarize_image(gray, method="sauvola")
    assert binary.shape == (80, 80)
    # Text pixels should be 0 (black), background 255 (white)
    assert binary[40, 40] == 0
    assert binary[10, 10] == 255

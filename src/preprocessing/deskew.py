"""
Deskewing module for straightening tilted or angled medical prescription scans.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.config import DESKEW_MAX_ANGLE
from src.preprocessing.enhancer import to_numpy_image


def calculate_skew_angle(gray_or_bin: np.ndarray, max_angle: float = DESKEW_MAX_ANGLE) -> float:
    """
    Calculates the dominant skew angle of handwritten/printed lines in degrees.
    Returns angle in degrees (negative for counterclockwise tilt, positive for clockwise).
    """
    if len(gray_or_bin.shape) != 2:
        gray = cv2.cvtColor(gray_or_bin, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bin

    # Invert binary image so text ink is white (255) on dark background (0)
    # Check if image is already binary
    unique_vals = np.unique(gray)
    if len(unique_vals) <= 2:
        # If background is white (255) and text is black (0), invert
        if np.mean(gray) > 127:
            binary = cv2.bitwise_not(gray)
        else:
            binary = gray
    else:
        # Apply Otsu binarization and invert
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find coordinates of all foreground text pixels
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return 0.0

    # minAreaRect computes minimum bounding box around text
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Convert OpenCV minAreaRect angle to orientation relative to horizontal
    if angle < -45.0:
        angle = -(90.0 + angle)
    elif angle > 45.0:
        angle = 90.0 - angle
    else:
        angle = -angle

    # Clamp extreme spurious angles
    if abs(angle) > max_angle:
        return 0.0

    return float(angle)


def deskew_image(
    image_input: Union[str, np.ndarray, Image.Image],
    border_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, float]:
    """
    Straightens tilted image and returns (deskewed_image, angle_corrected).
    """
    img = to_numpy_image(image_input)
    is_gray = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)

    if is_gray:
        gray = img if len(img.shape) == 2 else img[:, :, 0]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    angle = calculate_skew_angle(gray)

    # If angle is minimal (< 0.25 deg), avoid re-interpolation blur
    if abs(angle) < 0.25:
        return img, 0.0

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        img,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color if not is_gray else 255,
    )

    return deskewed, angle

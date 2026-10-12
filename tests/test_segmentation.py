"""
Unit tests for Segmentation Stage (BoundingBox, Line Detection, Aspect-Ratio Padding).
"""

import numpy as np
from PIL import Image
from src.segmentation.contour_detector import (
    BoundingBox,
    sort_bounding_boxes,
    segment_text_lines,
    crop_bounding_boxes,
)
from src.recognition.trocr_engine import pad_to_aspect_ratio


def test_bounding_box_properties():
    box = BoundingBox(x=10, y=20, w=100, h=30)
    assert box.x2 == 110
    assert box.y2 == 50
    assert box.area == 3000
    assert box.to_tuple() == (10, 20, 100, 30)


def test_sort_bounding_boxes_reading_order():
    # Box 1: top line, left
    b1 = BoundingBox(x=10, y=20, w=50, h=20)
    # Box 2: top line, right
    b2 = BoundingBox(x=100, y=22, w=50, h=20)
    # Box 3: second line, left
    b3 = BoundingBox(x=10, y=80, w=50, h=20)
    # Box 4: second line, right
    b4 = BoundingBox(x=100, y=82, w=50, h=20)

    # Scramble order
    scrambled = [b4, b1, b3, b2]
    sorted_boxes = sort_bounding_boxes(scrambled)

    # Expected order: b1, b2, b3, b4
    assert sorted_boxes[0] == b1
    assert sorted_boxes[1] == b2
    assert sorted_boxes[2] == b3
    assert sorted_boxes[3] == b4


def test_pad_to_aspect_ratio():
    # Test wide image (200x50)
    img = Image.new("RGB", (200, 50), (10, 20, 30))
    padded = pad_to_aspect_ratio(img, target_size=(384, 384))
    assert padded.size == (384, 384)

    # Verify background padding is white (255, 255, 255) at the corners
    corner_pixel = padded.getpixel((0, 0))
    assert corner_pixel == (255, 255, 255)

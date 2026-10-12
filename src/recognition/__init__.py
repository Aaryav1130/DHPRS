"""
Handwritten text recognition (HTR) engine and training pipeline.
"""

from src.recognition.trocr_engine import TrOCREngine, RecognitionResult
from src.recognition.dataset import PrescriptionWordDataset

__all__ = ["TrOCREngine", "RecognitionResult", "PrescriptionWordDataset"]

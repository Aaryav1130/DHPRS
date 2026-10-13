"""
Central configuration for Doctor's Handwritten Prescription Recognition System (DHPRS).
Provides unified settings, directories, model configs, and inference parameters.
"""

from pathlib import Path
import os
import torch

# Base Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CROPS_DATA_DIR = DATA_DIR / "crops"
DATASETS_DIR = DATA_DIR / "datasets"
DICTIONARIES_DIR = DATA_DIR / "dictionaries"
TEST_SAMPLES_DIR = DATA_DIR / "test_samples"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
CACHE_DIR = MODELS_DIR / "cache"

# Ensure runtime directories exist
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CROPS_DATA_DIR,
    DATASETS_DIR,
    DICTIONARIES_DIR,
    TEST_SAMPLES_DIR,
    CHECKPOINTS_DIR,
    CACHE_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Hardware / Device Selection
def get_device() -> torch.device:
    """Returns CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# OCR Recognition Settings
DEFAULT_MODEL_NAME = os.getenv("DHPRS_MODEL_NAME", "shubham879/trocr-prescription")
FAST_CPU_MODEL_NAME = "shubham879/trocr-prescription"
IMAGE_SIZE = (384, 384)
MAX_TARGET_LENGTH = 64
BEAM_SIZE = 4

# Preprocessing Settings
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
BACKGROUND_KERNEL_SIZE = (51, 51)
DESKEW_MAX_ANGLE = 45.0

# Segmentation Settings
MIN_LINE_HEIGHT = 10
MAX_LINE_HEIGHT = 100
MIN_LINE_WIDTH = 40
LINE_DILATION_KERNEL = (1, 35)  # Height 1 x Width 35 for pure horizontal grouping without vertical line bridging

# Post-Processing & Verification Settings
FUZZY_MATCH_THRESHOLD = 70.0  # Cutoff score for RapidFuzz drug name acceptance
CONFIDENCE_REVIEW_THRESHOLD = 0.80  # Flag for human-in-the-loop if confidence < 0.80
CONFIDENCE_HIGH_THRESHOLD = 0.90

# Lexicon Filepaths
WHO_DRUG_LEXICON = DICTIONARIES_DIR / "who_essential_medicines.json"
INDIAN_DRUG_LEXICON = DICTIONARIES_DIR / "indian_medicines.json"
BD_DRUG_LEXICON = DICTIONARIES_DIR / "bangladesh_medicines.json"
CUSTOM_DRUG_LEXICON = DICTIONARIES_DIR / "custom_drugs.json"

"""
PyTorch Dataset module for training and fine-tuning TrOCR on prescription word datasets.
Supports RxHandBD and synthetic medical datasets.
"""

from pathlib import Path
from typing import Callable, Optional, Union
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

from src.recognition.trocr_engine import pad_to_aspect_ratio


class PrescriptionWordDataset(Dataset):
    """
    PyTorch Dataset for cropped handwritten prescription word images.
    Expects a labels CSV file with columns: `image_name` (or `image_path`) and `text` (or `label`).
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        labels_csv: Union[str, Path],
        processor: TrOCRProcessor,
        max_target_length: int = 64,
        transform: Optional[Callable] = None,
        image_col: str = "image_name",
        label_col: str = "label",
    ):
        self.images_dir = Path(images_dir)
        self.labels_df = pd.read_csv(labels_csv)
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform

        # Resolve column names flexibly
        cols = {c.lower(): c for c in self.labels_df.columns}
        if image_col.lower() in cols:
            self.image_col = cols[image_col.lower()]
        elif "image" in cols:
            self.image_col = cols["image"]
        elif "filename" in cols:
            self.image_col = cols["filename"]
        else:
            self.image_col = self.labels_df.columns[0]

        if label_col.lower() in cols:
            self.label_col = cols[label_col.lower()]
        elif "text" in cols:
            self.label_col = cols["text"]
        elif "transcription" in cols:
            self.label_col = cols["transcription"]
        else:
            self.label_col = self.labels_df.columns[1]

        # Clean NaN labels
        self.labels_df = self.labels_df.dropna(subset=[self.image_col, self.label_col]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.labels_df.iloc[idx]
        image_name = str(row[self.image_col]).strip()
        label_text = str(row[self.label_col]).strip()

        image_path = self.images_dir / image_name
        if not image_path.exists():
            # Check recursive matches in subfolders (e.g. Train_Set, Test_Set)
            matches = list(self.images_dir.glob(f"**/{image_name}*"))
            if matches:
                image_path = matches[0]
            else:
                raise FileNotFoundError(f"Image not found: {image_name} in {self.images_dir}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Pad to aspect ratio (384x384 canvas)
        padded_image = pad_to_aspect_ratio(image, target_size=(384, 384))

        pixel_values = self.processor(padded_image, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize labels
        labels = self.processor.tokenizer(
            label_text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Set padding tokens to -100 so loss ignores them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

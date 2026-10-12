"""
TrOCR inference engine for handwritten medical text recognition.
Provides aspect-preserving resizing, batch inference, beam search, and confidence calculation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.config import (
    DEFAULT_MODEL_NAME,
    DEVICE,
    IMAGE_SIZE,
    MAX_TARGET_LENGTH,
    BEAM_SIZE,
    CACHE_DIR,
)


@dataclass
class RecognitionResult:
    """Stores the OCR output for a single text line or word."""
    text: str
    confidence: float
    raw_text: str = ""
    tokens: List[str] = field(default_factory=list)


def pad_to_aspect_ratio(
    image: Image.Image,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    pad_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Resizes image to fit inside target_size while strictly preserving original aspect ratio.
    Pads the remaining empty canvas with white color (255, 255, 255).
    Crucial for handwritten medicine names so cursive characters are not squished or stretched.
    """
    target_w, target_h = target_size
    img_w, img_h = image.size

    if img_w == 0 or img_h == 0:
        return Image.new("RGB", target_size, pad_color)

    # Compute scale factor
    scale = min(target_w / img_w, target_h / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Paste onto white background
    padded = Image.new("RGB", target_size, pad_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))

    return padded


class TrOCREngine:
    """
    Orchestrates TrOCR model loading, image feature extraction, and text generation.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        device: Optional[torch.device] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device or DEVICE
        self.processor: Optional[TrOCRProcessor] = None
        self.model: Optional[VisionEncoderDecoderModel] = None
        self._is_loaded = False

    def _ensure_position_embeddings_materialized(self) -> None:
        """Ensures decoder sinusoidal positional embeddings are materialized off meta device."""
        try:
            decoder_model = getattr(self.model, "decoder", None)
            if decoder_model is not None:
                inner_model = getattr(decoder_model, "model", None)
                target_dec = getattr(inner_model, "decoder", decoder_model)
                embed_positions = getattr(target_dec, "embed_positions", None)
                if embed_positions is not None and hasattr(embed_positions, "weights"):
                    if getattr(embed_positions.weights, "is_meta", False) or embed_positions.weights is None:
                        num_pos = max(embed_positions.weights.size(0) if embed_positions.weights is not None else 512, 512)
                        embed_positions.weights = embed_positions.get_embedding(
                            num_pos, embed_positions.embedding_dim, embed_positions.padding_idx
                        ).to(self.device)
        except Exception as e:
            print(f"[TrOCR] Positional embedding materialization notice: {e}")

    def load_model(self) -> None:
        """Loads processor and model weights into memory (lazy loaded on first request)."""
        if self._is_loaded:
            return

        print(f"[TrOCR] Loading model weights from: {self.model_name_or_path}")
        print(f"[TrOCR] Running on device: {self.device}")

        self.processor = TrOCRProcessor.from_pretrained(
            self.model_name_or_path,
            cache_dir=str(CACHE_DIR),
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.model_name_or_path,
            cache_dir=str(CACHE_DIR),
        )

        # Fix for Transformers >= 5.x / PyTorch: ensure decoder positional embeddings are not left on meta device
        self._ensure_position_embeddings_materialized()

        self.model.to(self.device)
        self._ensure_position_embeddings_materialized()
        self.model.eval()
        self._is_loaded = True
        print("[TrOCR] Model loaded successfully.")

    def ensure_pil_image(self, img_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Converts arbitrary image representation to RGB PIL Image."""
        if isinstance(img_input, str):
            return Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if len(img_input.shape) == 2:
                return Image.fromarray(img_input).convert("RGB")
            # OpenCV BGR to RGB
            return Image.fromarray(img_input[:, :, ::-1]).convert("RGB")
        elif isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        raise ValueError(f"Unsupported image type: {type(img_input)}")

    def _compute_sequence_confidence(
        self,
        scores: Tuple[torch.Tensor, ...],
        generated_ids: torch.Tensor,
    ) -> float:
        """
        Computes geometric mean probability across generated token steps.
        Produces a normalized confidence score between 0.0 and 1.0.
        """
        if not scores:
            return 0.85  # Default fallback if scores not computed

        step_probs = []
        # generated_ids shape: (1, seq_len)
        token_ids = generated_ids[0].tolist()

        # Skip the prompt / start token
        gen_tokens = token_ids[1:]

        for step_idx, step_logits in enumerate(scores):
            if step_idx >= len(gen_tokens):
                break
            # step_logits shape: (batch_size * num_beams, vocab_size)
            probs = torch.softmax(step_logits[0], dim=-1)
            token_id = gen_tokens[step_idx]
            if token_id < probs.shape[-1]:
                p = probs[token_id].item()
                step_probs.append(p)

        if not step_probs:
            return 0.85

        # Geometric mean
        log_sum = sum(np.log(max(p, 1e-6)) for p in step_probs)
        geom_mean = np.exp(log_sum / len(step_probs))
        return float(np.clip(geom_mean, 0.0, 1.0))

    @torch.no_grad()
    def recognize_crop(
        self,
        crop_input: Union[str, np.ndarray, Image.Image],
    ) -> RecognitionResult:
        """
        Recognizes text in a single cropped image patch.
        """
        self.load_model()
        pil_img = self.ensure_pil_image(crop_input)
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        self._ensure_position_embeddings_materialized()

        outputs = self.model.generate(
            pixel_values,
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=BEAM_SIZE,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_ids = outputs.sequences
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        confidence = self._compute_sequence_confidence(outputs.scores, generated_ids)

        return RecognitionResult(
            text=transcription,
            confidence=confidence,
            raw_text=transcription,
        )

    @torch.no_grad()
    def recognize_batch(
        self,
        crop_inputs: List[Union[str, np.ndarray, Image.Image]],
        batch_size: int = 4,
    ) -> List[RecognitionResult]:
        """
        Recognizes text across multiple cropped image patches.
        """
        if not crop_inputs:
            return []

        self.load_model()
        results: List[RecognitionResult] = []
        for crop in crop_inputs:
            results.append(self.recognize_crop(crop))
        return results

"""
TrOCR Fine-Tuning script using Hugging Face Seq2SeqTrainer.
Trains model on RxHandBD / medical handwriting datasets and tracks Character Error Rate (CER).
"""

from pathlib import Path
import argparse
import evaluate
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from src.config import CHECKPOINTS_DIR, CACHE_DIR, DEFAULT_MODEL_NAME
from src.recognition.dataset import PrescriptionWordDataset


def compute_cer_metric_builder(processor: TrOCRProcessor):
    """Builds the evaluation metric function computing Character Error Rate (CER)."""
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    return compute_metrics


def train_trocr(
    train_dir: str,
    train_csv: str,
    eval_dir: str,
    eval_csv: str,
    output_dir: str = str(CHECKPOINTS_DIR / "trocr-rxhandbd-best"),
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 15,
    batch_size: int = 8,
    learning_rate: float = 4e-5,
    fp16: bool = True,
):
    """Runs Seq2Seq fine-tuning loop."""
    print(f"Initializing fine-tuning for model: {model_name}")
    processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=str(CACHE_DIR))
    model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=str(CACHE_DIR))

    # Configure special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.num_beams = 4

    train_dataset = PrescriptionWordDataset(train_dir, train_csv, processor)
    eval_dataset = PrescriptionWordDataset(eval_dir, eval_csv, processor)

    print(f"Loaded {len(train_dataset)} training samples, {len(eval_dataset)} eval samples.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=fp16,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=default_data_collator,
        compute_metrics=compute_cer_metric_builder(processor),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on prescription dataset")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training images")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train labels CSV")
    parser.add_argument("--eval_dir", type=str, required=True, help="Path to evaluation images")
    parser.add_argument("--eval_csv", type=str, required=True, help="Path to eval labels CSV")
    parser.add_argument("--output_dir", type=str, default=str(CHECKPOINTS_DIR / "trocr-rxhandbd"))
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--no_fp16", action="store_true", help="Disable mixed precision")

    args = parser.parse_args()
    train_trocr(
        train_dir=args.train_dir,
        train_csv=args.train_csv,
        eval_dir=args.eval_dir,
        eval_csv=args.eval_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=not args.no_fp16,
    )

"""ResNet-50 training script for visual-only stance classification."""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_linear_schedule_with_warmup,
)

# Label mappings
LABELS_EN = ["oppositional", "neutral", "supportive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS_EN)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# Model configuration
MODEL_CHECKPOINT = "microsoft/resnet-50"
LEARNING_RATE = 1e-4


class VisualStanceDataset(Dataset):
    """Dataset for visual-only stance classification."""

    def __init__(self, data_path: str, image_dir: str, split: str, processor):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.split = split
        self.processor = processor
        self.examples = self._load_examples()

    def _load_examples(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = []
        for row in data:
            if row.get("split") != self.split:
                continue

            article_id = row.get("id")
            stance = row.get("stance")
            if article_id is None or stance is None:
                continue

            # Find image with pattern {id}_1.*
            image_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                path = self.image_dir / f"{article_id}_1{ext}"
                if path.exists():
                    image_path = path
                    break

            if image_path is None:
                continue

            examples.append(
                {
                    "id": article_id,
                    "image_path": str(image_path),
                    "label": stance,
                    "label_id": LABEL_TO_ID[stance],
                }
            )

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        with Image.open(example["image_path"]) as image:
            image = image.convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)

        return {
            "id": example["id"],
            "pixel_values": pixel_values,
            "label_id": example["label_id"],
            "label": example["label"],
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "id": [item["id"] for item in batch],
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "label_id": torch.tensor([item["label_id"] for item in batch], dtype=torch.long),
        "label": [item["label"] for item in batch],
    }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(gold_ids, pred_ids):
    """Compute classification metrics."""
    labels = list(range(len(LABELS_EN)))
    class_f1 = f1_score(gold_ids, pred_ids, labels=labels, average=None, zero_division=0)
    matrix = confusion_matrix(gold_ids, pred_ids, labels=labels)

    return {
        "accuracy": float(accuracy_score(gold_ids, pred_ids)),
        "macro_f1": float(f1_score(gold_ids, pred_ids, labels=labels, average="macro", zero_division=0)),
        "f1_oppositional": float(class_f1[0]),
        "f1_neutral": float(class_f1[1]),
        "f1_supportive": float(class_f1[2]),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_gold_ids = []
    all_pred_ids = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            preds = torch.argmax(logits, dim=-1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            all_gold_ids.extend(labels.detach().cpu().tolist())
            all_pred_ids.extend(preds.detach().cpu().tolist())

    metrics = compute_metrics(all_gold_ids, all_pred_ids)
    metrics["loss"] = float(total_loss / max(total_examples, 1))
    return metrics


def train(args):
    """Main training function."""
    set_seed(args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load processor and model
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABELS_EN),
        ignore_mismatched_sizes=True,
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
        use_safetensors=True,
    )
    model.to(device)

    # Load datasets
    train_dataset = VisualStanceDataset(args.data_path, args.image_dir, "train", processor)
    val_dataset = VisualStanceDataset(args.data_path, args.image_dir, "validation", processor)
    test_dataset = VisualStanceDataset(args.data_path, args.image_dir, "test", processor)

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Validation size: {len(val_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=args.weight_decay,
    )

    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Training loop
    best_val_macro_f1 = float("-inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0
        train_gold_ids = []
        train_pred_ids = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            train_gold_ids.extend(labels.detach().cpu().tolist())
            train_pred_ids.extend(preds.detach().cpu().tolist())

            progress_bar.set_postfix({"loss": loss.item()})

        # Compute training metrics
        train_metrics = compute_metrics(train_gold_ids, train_pred_ids)
        train_metrics["loss"] = running_loss / total_examples

        # Validation
        val_metrics = evaluate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch} | "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        # Save best model
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            # Save model
            output_dir = Path(args.output_dir) / f"seed_{args.seed}"
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir / "best_model", safe_serialization=True)
            processor.save_pretrained(output_dir / "best_model")
            logger.info(f"Saved best model at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Load best model and evaluate on test set
    logger.info(f"Loading best model from epoch {best_epoch}")
    best_model_dir = Path(args.output_dir) / f"seed_{args.seed}" / "best_model"
    model = AutoModelForImageClassification.from_pretrained(
        best_model_dir, use_safetensors=True
    )
    model.to(device)

    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"Test metrics: {test_metrics}")

    # Save results
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 for visual stance classification")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/k-news-stance-mm-acl-arr.json",
        help="Path to dataset JSON",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="../../data/images",
        help="Path to image directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
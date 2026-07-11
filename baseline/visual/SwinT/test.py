"""SwinV2-Base testing script for visual-only stance classification."""

import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Label mappings
LABELS_EN = ["oppositional", "neutral", "supportive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS_EN)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


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


def compute_metrics(gold_ids, pred_ids):
    """Compute classification metrics."""
    labels = list(range(len(LABELS_EN)))
    class_f1 = f1_score(gold_ids, pred_ids, labels=labels, average=None, zero_division=0)
    matrix = confusion_matrix(gold_ids, pred_ids, labels=labels)

    return {
        "accuracy": float(accuracy_score(gold_ids, pred_ids)),
        "macro_f1": float(
            f1_score(gold_ids, pred_ids, labels=labels, average="macro", zero_division=0)
        ),
        "f1_oppositional": float(class_f1[0]),
        "f1_neutral": float(class_f1[1]),
        "f1_supportive": float(class_f1[2]),
        "confusion_matrix": matrix.tolist(),
    }


def test(args):
    """Main testing function."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load processor and model
    model_dir = Path(args.model_dir)
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(
        model_dir, use_safetensors=True
    )
    model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = VisualStanceDataset(args.data_path, args.image_dir, "test", processor)
    logger.info(f"Test size: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Evaluation
    all_gold_ids = []
    all_pred_ids = []
    all_predictions = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            preds = torch.argmax(logits, dim=-1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            gold_ids = labels.detach().cpu().tolist()
            pred_ids = preds.detach().cpu().tolist()
            all_gold_ids.extend(gold_ids)
            all_pred_ids.extend(pred_ids)

            # Save predictions
            for i, article_id in enumerate(batch["id"]):
                all_predictions.append(
                    {
                        "id": article_id,
                        "prediction": ID_TO_LABEL[pred_ids[i]],
                        "gold": batch["label"][i],
                    }
                )

    # Compute metrics
    metrics = compute_metrics(all_gold_ids, all_pred_ids)
    metrics["loss"] = float(total_loss / max(total_examples, 1))

    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"  F1 Oppositional: {metrics['f1_oppositional']:.4f}")
    logger.info(f"  F1 Neutral: {metrics['f1_neutral']:.4f}")
    logger.info(f"  F1 Supportive: {metrics['f1_supportive']:.4f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "predictions.jsonl", "w") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info(f"Results saved to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test SwinV2-Base for visual stance classification")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
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
        default="./test_output",
        help="Output directory",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()
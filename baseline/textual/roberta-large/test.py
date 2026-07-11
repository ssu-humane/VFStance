"""RoBERTa-Large testing script for textual stance classification."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# Label mappings
LABELS_EN = ["oppositional", "neutral", "supportive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS_EN)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def load_data_from_json(file_path):
    """Load dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text examples."""
    sep_token = tokenizer.sep_token or "</s>"
    texts = [
        f"{issue} {sep_token} {headline} {sep_token} {article}"
        for issue, headline, article in zip(examples['issue'], examples['headline'], examples['article'])
    ]

    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
    )


def compute_metrics(gold_ids, pred_ids):
    """Compute classification metrics."""
    labels = list(range(len(LABELS_EN)))

    accuracy = accuracy_score(gold_ids, pred_ids)
    c_report = classification_report(gold_ids, pred_ids, output_dict=True, zero_division=0)
    matrix = confusion_matrix(gold_ids, pred_ids, labels=labels)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(c_report['macro avg']['f1-score']),
        "macro_precision": float(c_report['macro avg']['precision']),
        "macro_recall": float(c_report['macro avg']['recall']),
        "f1_oppositional": float(c_report.get('0', {'f1-score': 0})['f1-score']),
        "f1_neutral": float(c_report.get('1', {'f1-score': 0})['f1-score']),
        "f1_supportive": float(c_report.get('2', {'f1-score': 0})['f1-score']),
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

    # Load model and tokenizer
    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load data
    data = load_data_from_json(args.data_path)

    # Prepare test dataset
    test_data = []
    for item in data:
        if item.get('split') != 'test':
            continue

        article_id = item.get('id')
        headline = item.get('headline', '')
        article = item.get('article', '')
        issue = item.get('issue', '')
        stance = item.get('stance')

        if stance is None:
            continue

        label_id = LABEL_TO_ID[stance]

        test_data.append({
            'id': article_id,
            'issue': issue,
            'headline': headline,
            'article': article,
            'label': label_id
        })

    logger.info(f"Test size: {len(test_data)}")

    # Create dataset
    test_dataset = Dataset.from_list(test_data)

    # Tokenize dataset
    tokenized_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        num_proc=4,
        desc="Tokenizing test dataset"
    )

    # Remove unnecessary columns
    columns_to_remove = ['id', 'article', 'issue', 'headline']
    current_columns = tokenized_dataset.column_names
    remove = [col for col in columns_to_remove if col in current_columns]
    if remove:
        tokenized_dataset = tokenized_dataset.remove_columns(remove)

    # Create dataloader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False
    )

    # Evaluation
    all_predictions = []
    all_gold_ids = []
    all_pred_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_gold_ids.extend(labels.cpu().numpy())
            all_pred_ids.extend(preds.cpu().numpy())

    # Compute metrics
    metrics = compute_metrics(all_gold_ids, all_pred_ids)

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

    # Save predictions
    predictions_data = []
    for i, (gold, pred) in enumerate(zip(all_gold_ids, all_pred_ids)):
        predictions_data.append({
            "index": i,
            "gold": ID_TO_LABEL[gold],
            "prediction": ID_TO_LABEL[pred]
        })

    with open(output_dir / "predictions.jsonl", "w") as f:
        for pred in predictions_data:
            f.write(json.dumps(pred) + "\n")

    logger.info(f"Results saved to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test RoBERTa-Large for textual stance classification")
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
        "--output_dir",
        type=str,
        default="./test_output",
        help="Output directory",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()

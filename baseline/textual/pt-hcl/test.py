"""PT-HCL testing script for the complete pipeline."""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABELS = ["oppositional", "neutral", "supportive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


class HCLModel(nn.Module):
    """Hierarchical Contrastive Learning Model for inference."""

    def __init__(self, encoder_path, num_classes, hidden_dim=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )

    def encode(self, input_ids, attention_mask):
        """Encode input to get representations."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output

    def forward(self, input_ids, attention_mask):
        # Encode input
        repr = self.encode(input_ids, attention_mask)
        # Classification
        dropped = self.dropout(repr)
        logits = self.classifier(dropped)
        return logits


class TestDataset(Dataset):
    """Dataset for testing."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        issue = item.get('issue', '')
        headline = item.get('headline', '')
        article = item.get('article', '')
        stance = item.get('stance', '')

        # Create input text
        sep_token = self.tokenizer.sep_token or "</s>"
        text = f"{issue} {sep_token} {headline} {sep_token} {article}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Map stance to ID
        stance_id = LABEL_TO_ID.get(stance, LABEL_TO_ID["neutral"])

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(stance_id, dtype=torch.long),
            'id': item.get('id', idx)
        }


def test(args):
    """Main testing function."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model components
    model_dir = Path(args.model_dir)

    # Check which stage to use
    if args.stage == "vanilla":
        model_path = model_dir / "1_vanilla_model" / "vanilla_output" / f"seed_{args.seed}" / "best_model"
        label_mapping_path = model_dir / "1_vanilla_model" / "vanilla_output" / f"seed_{args.seed}" / "label_mapping.json"
    elif args.stage == "hcl":
        hcl_dir = model_dir / "4_hcl" / "hcl_output" / f"seed_{args.seed}"
        model_checkpoint = torch.load(hcl_dir / "best_hcl_model.pt", map_location=device)

        # Load pretrained encoder path
        pretext_dir = model_dir / "3_pt" / "pretext_output" / f"seed_{args.seed}"
        encoder_path = str(pretext_dir / "pretrained_encoder")

        # Load label mapping
        label_mapping_path = model_dir / "1_vanilla_model" / "vanilla_output" / f"seed_{args.seed}" / "label_mapping.json"
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    # Load label mapping
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

    # Load test data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter test split
    test_data = [item for item in data if item.get('split') == 'test']
    print(f"Test size: {len(test_data)}")

    if args.stage == "vanilla":
        # Use vanilla model directly from transformers
        from transformers import AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:  # hcl
        # Use HCL model
        tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        model = HCLModel(encoder_path, len(label_mapping))
        model.load_state_dict(model_checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    # Create dataset and dataloader
    test_dataset = TestDataset(test_data, tokenizer, args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluation
    all_preds = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            ids = batch['id']

            if args.stage == "vanilla":
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            else:  # hcl
                logits = model(input_ids, attention_mask)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_ids.extend(ids)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # Create label names for report
    label_names = LABELS
    report = classification_report(all_labels, all_preds, target_names=label_names)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Results ({args.stage.upper()} stage):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"\nClassification Report:")
    print(report)
    print(f"\nConfusion Matrix:")
    print(conf_matrix)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'stage': args.stage,
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
    }

    with open(output_dir / f"test_results_{args.stage}.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save predictions
    predictions_data = []
    for article_id, gold, pred in zip(all_ids, all_labels, all_preds):
        predictions_data.append({
            "id": int(article_id) if isinstance(article_id, (np.integer, torch.Tensor)) else article_id,
            "gold": label_names[gold],
            "prediction": label_names[pred]
        })

    with open(output_dir / f"predictions_{args.stage}.jsonl", "w", encoding='utf-8') as f:
        for pred_item in predictions_data:
            f.write(json.dumps(pred_item, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Test PT-HCL models")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=".",
        help="Base directory containing all PT-HCL stages"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["vanilla", "hcl"],
        default="hcl",
        help="Which stage model to test"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/k-news-stance-mm-acl-arr.json",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_output",
        help="Output directory for results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Model seed to load")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()

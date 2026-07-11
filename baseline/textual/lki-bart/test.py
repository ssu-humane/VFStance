"""LKI-BART testing script for textual stance classification."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

# Label mappings
LABELS_EN = ["oppositional", "neutral", "supportive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS_EN)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

STANCE_PLACEHOLDER = "<stance>"


class StanceDataset(Dataset):
    """Dataset for stance classification with BART."""

    def __init__(self, data, tokenizer, max_seq_length, max_target_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        issue = item['issue']
        article = item['article']
        knowledge = item.get('knowledge', '')
        stance_id = item['stance_id']

        # Create input with knowledge
        if knowledge:
            input_text = f"{issue}: {article} [지식] {knowledge}"
        else:
            input_text = f"{issue}: {article}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'stance_id': stance_id,
            'id': item['id']
        }


class StanceHead(nn.Module):
    """Stance classification head with projection."""

    def __init__(self, hidden_dim, projection_dim, num_classes):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, projection_dim)
        self.classifier = nn.Linear(projection_dim, num_classes)
        self.prototypes = nn.Parameter(torch.randn(num_classes, projection_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, hidden_states):
        projected = self.projection(hidden_states)
        logits = self.classifier(projected)
        return logits, projected


class LKI_BART(nn.Module):
    """LKI-BART model with stance head."""

    def __init__(self, model_name, projection_dim, num_classes):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        self.stance_head = StanceHead(
            self.bart.config.hidden_size,
            projection_dim,
            num_classes
        )

    def forward(self, input_ids, attention_mask):
        # BART forward pass (encoder only for inference)
        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get encoder hidden states
        encoder_hidden = encoder_outputs.last_hidden_state
        pooled_hidden = encoder_hidden.mean(dim=1)

        # Stance classification
        stance_logits, projected = self.stance_head(pooled_hidden)

        return stance_logits


def compute_metrics(gold_ids, pred_ids):
    """Compute classification metrics."""
    labels = list(range(len(LABELS_EN)))

    accuracy = accuracy_score(gold_ids, pred_ids)
    macro_f1 = f1_score(gold_ids, pred_ids, average='macro')
    report = classification_report(gold_ids, pred_ids, output_dict=True, zero_division=0)
    matrix = confusion_matrix(gold_ids, pred_ids, labels=labels)

    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'f1_oppositional': float(report.get('0', {'f1-score': 0})['f1-score']),
        'f1_neutral': float(report.get('1', {'f1-score': 0})['f1-score']),
        'f1_supportive': float(report.get('2', {'f1-score': 0})['f1-score']),
        'confusion_matrix': matrix.tolist(),
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

    # Load config
    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json", 'r') as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

    # Initialize and load model
    model = LKI_BART(
        config['model_name'],
        config['projection_dim'],
        config['num_classes']
    )
    model.bart.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Load data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load knowledge if available
    knowledge_mapping = {}
    if args.knowledge_path:
        with open(args.knowledge_path, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
            for item in knowledge_data:
                knowledge_mapping[item.get('id')] = item.get('llm_knowledge', '')

    # Prepare test dataset
    test_data = []
    for item in data:
        if item.get('split') != 'test':
            continue

        article_id = item.get('id')
        issue = item.get('issue', '')
        article = item.get('article', '')
        stance = item.get('stance')

        if stance is None:
            continue

        test_data.append({
            'id': article_id,
            'issue': issue,
            'article': article,
            'knowledge': knowledge_mapping.get(article_id, ''),
            'stance': stance,
            'stance_id': LABEL_TO_ID[stance]
        })

    logger.info(f"Test size: {len(test_data)}")

    # Create dataset and dataloader
    test_dataset = StanceDataset(test_data, tokenizer, args.max_seq_length, args.max_target_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluation
    all_preds = []
    all_golds = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            stance_ids = batch['stance_id']
            article_ids = batch['id']

            # Forward pass
            stance_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predictions
            preds = torch.argmax(stance_logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_golds.extend(stance_ids.numpy())
            all_ids.extend(article_ids)

    # Compute metrics
    metrics = compute_metrics(all_golds, all_preds)

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
    for article_id, gold, pred in zip(all_ids, all_golds, all_preds):
        predictions_data.append({
            "id": article_id,
            "gold": ID_TO_LABEL[gold],
            "prediction": ID_TO_LABEL[pred]
        })

    with open(output_dir / "predictions.jsonl", "w") as f:
        for pred_item in predictions_data:
            f.write(json.dumps(pred_item) + "\n")

    logger.info(f"Results saved to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test LKI-BART for textual stance classification")
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
        "--knowledge_path",
        type=str,
        default=None,
        help="Path to knowledge dataset JSON (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_output",
        help="Output directory",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max_target_length", type=int, default=64, help="Max target length")

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()
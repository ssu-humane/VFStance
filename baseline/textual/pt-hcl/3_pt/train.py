"""PT-HCL Stage 3: Pretext task training with updated data format."""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import accuracy_score, f1_score

FROZEN_LAYER_COUNT = 7


def set_seed(seed_value):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PretextDataset(Dataset):
    """Dataset for pretext task (sentence position prediction)."""

    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.articles = self._group_by_article()

    def _group_by_article(self):
        """Group sentences by article."""
        articles = {}
        for _, row in self.df.iterrows():
            article_id = row['article_id']
            if article_id not in articles:
                articles[article_id] = {
                    'sentences': [],
                    'issue': row['issue'],
                    'headline': row.get('headline', ''),
                    'stance_label': row['stance_label']
                }
            articles[article_id]['sentences'].append({
                'text': row['sentence'],
                'index': row['sentence_index']
            })

        # Sort sentences by index
        for article_id in articles:
            articles[article_id]['sentences'].sort(key=lambda x: x['index'])

        return list(articles.values())

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        sentences = article['sentences']

        if len(sentences) < 2:
            # If only one sentence, duplicate it
            sent1 = sentences[0]['text']
            sent2 = sentences[0]['text']
            label = 1  # Same position
        else:
            # Sample two different sentences
            idx1, idx2 = random.sample(range(len(sentences)), 2)
            sent1 = sentences[idx1]['text']
            sent2 = sentences[idx2]['text']

            # Label: 1 if sent1 comes before sent2, 0 otherwise
            label = 1 if idx1 < idx2 else 0

        # Combine with issue and headline
        issue = article['issue']
        headline = article.get('headline', '')
        sep_token = self.tokenizer.sep_token or "</s>"
        text1 = f"{issue} {sep_token} {headline} {sep_token} {sent1}"
        text2 = f"{issue} {sep_token} {headline} {sep_token} {sent2}"

        # Tokenize
        encoding = self.tokenizer(
            text1, text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class PretextModel(nn.Module):
    """Model for pretext task."""

    def __init__(self, model_name, num_classes=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def freeze_lower_roberta_layers(model, frozen_layer_count):
    """Freeze the bottom encoder layers of a RoBERTa encoder."""
    for layer in model.encoder.encoder.layer[:frozen_layer_count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


def train(args):
    """Main training function."""
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data from vanilla model output
    vanilla_output = Path(args.vanilla_output_dir) / f"seed_{args.seed}"
    train_df = pd.read_csv(vanilla_output / "train_expanded.csv")
    val_df = pd.read_csv(vanilla_output / "val_expanded.csv")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = PretextModel(args.model_name)
    freeze_lower_roberta_layers(model, FROZEN_LAYER_COUNT)
    model.to(device)

    # Create datasets and dataloaders
    train_dataset = PretextDataset(train_df, tokenizer, args.max_length)
    val_dataset = PretextDataset(val_df, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    best_val_accuracy = 0
    patience_counter = 0
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0

            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_acc
            }, output_dir / "best_pretext_model.pt")

            # Save encoder only for downstream task
            model.encoder.save_pretrained(output_dir / "pretrained_encoder")
            tokenizer.save_pretrained(output_dir / "pretrained_encoder")

            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save training results
    results = {
        'best_val_accuracy': best_val_accuracy,
        'final_epoch': epoch
    }
    with open(output_dir / "pretext_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
    return best_val_accuracy


def main():
    parser = argparse.ArgumentParser(description="PT-HCL Stage 3: Pretext Task Training")
    parser.add_argument("--vanilla_output_dir", type=str, default="../1_vanilla_model/vanilla_output")
    parser.add_argument("--model_name", type=str, default="klue/roberta-large")
    parser.add_argument("--output_dir", type=str, default="./pretext_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

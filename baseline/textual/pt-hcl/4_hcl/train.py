"""PT-HCL Stage 4: Hierarchical Contrastive Learning with updated data format."""

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
from sklearn.metrics import accuracy_score, f1_score, classification_report

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


class HCLDataset(Dataset):
    """Dataset for Hierarchical Contrastive Learning."""

    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stance_to_examples = self._group_by_stance()

    def _group_by_stance(self):
        """Group examples by stance label."""
        stance_groups = {}
        for idx, row in self.df.iterrows():
            stance = row['stance_label']
            if stance not in stance_groups:
                stance_groups[stance] = []
            stance_groups[stance].append(idx)
        return stance_groups

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        issue = row['issue']
        headline = row.get('headline', '')
        sentence = row['sentence']
        stance_label = row['stance_label']

        # Get positive example (same stance)
        positive_indices = [i for i in self.stance_to_examples[stance_label] if i != idx]
        if positive_indices:
            pos_idx = random.choice(positive_indices)
            pos_row = self.df.iloc[pos_idx]
            pos_sentence = pos_row['sentence']
        else:
            pos_sentence = sentence  # Use same sentence if no other examples

        # Get negative example (different stance)
        negative_stances = [s for s in self.stance_to_examples.keys() if s != stance_label]
        if negative_stances:
            neg_stance = random.choice(negative_stances)
            neg_idx = random.choice(self.stance_to_examples[neg_stance])
            neg_row = self.df.iloc[neg_idx]
            neg_sentence = neg_row['sentence']
        else:
            neg_sentence = sentence  # Fallback

        # Combine with issue and headline
        sep_token = self.tokenizer.sep_token or "</s>"
        anchor_text = f"{issue} {sep_token} {headline} {sep_token} {sentence}"
        positive_text = f"{issue} {sep_token} {headline} {sep_token} {pos_sentence}"
        negative_text = f"{issue} {sep_token} {headline} {sep_token} {neg_sentence}"

        # Tokenize
        anchor_encoding = self.tokenizer(
            anchor_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        positive_encoding = self.tokenizer(
            positive_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        negative_encoding = self.tokenizer(
            negative_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(),
            'negative_input_ids': negative_encoding['input_ids'].squeeze(),
            'negative_attention_mask': negative_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(stance_label, dtype=torch.long)
        }


class HCLModel(nn.Module):
    """Hierarchical Contrastive Learning Model."""

    def __init__(self, encoder_path, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        hidden_dim = self.encoder.config.hidden_size

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

    def forward(self, anchor_ids, anchor_mask, positive_ids=None, positive_mask=None,
                negative_ids=None, negative_mask=None):
        # Encode anchor
        anchor_repr = self.encode(anchor_ids, anchor_mask)

        # Classification
        dropped_anchor = self.dropout(anchor_repr)
        logits = self.classifier(dropped_anchor)

        if positive_ids is not None and negative_ids is not None:
            # Encode positive and negative for contrastive learning
            positive_repr = self.encode(positive_ids, positive_mask)
            negative_repr = self.encode(negative_ids, negative_mask)

            # Project for contrastive learning
            anchor_proj = self.projection(anchor_repr)
            positive_proj = self.projection(positive_repr)
            negative_proj = self.projection(negative_repr)

            return logits, anchor_proj, positive_proj, negative_proj

        return logits


def freeze_lower_roberta_layers(model, frozen_layer_count):
    """Freeze the bottom encoder layers of a RoBERTa encoder."""
    for layer in model.encoder.encoder.layer[:frozen_layer_count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def hierarchical_contrastive_loss(anchor, positive, negative, labels, temperature=0.5, lambda_weight=0.5):
    """Compute hierarchical contrastive loss."""
    batch_size = anchor.size(0)

    # Normalize embeddings
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)

    # Compute similarities
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature
    neg_sim = torch.sum(anchor * negative, dim=1) / temperature

    # Basic contrastive loss
    basic_loss = -torch.mean(pos_sim - torch.log(torch.exp(pos_sim) + torch.exp(neg_sim)))

    # Hierarchical component: penalize more for confusing similar stances
    # This is simplified - in full implementation, would use stance hierarchy
    hierarchical_penalty = torch.mean(torch.relu(neg_sim - pos_sim + 0.5))

    total_loss = basic_loss + lambda_weight * hierarchical_penalty
    return total_loss


def train_epoch(model, dataloader, optimizer, device, num_classes):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_cl_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        anchor_ids = batch['anchor_input_ids'].to(device)
        anchor_mask = batch['anchor_attention_mask'].to(device)
        positive_ids = batch['positive_input_ids'].to(device)
        positive_mask = batch['positive_attention_mask'].to(device)
        negative_ids = batch['negative_input_ids'].to(device)
        negative_mask = batch['negative_attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logits, anchor_proj, positive_proj, negative_proj = model(
            anchor_ids, anchor_mask, positive_ids, positive_mask,
            negative_ids, negative_mask
        )

        # Classification loss
        ce_loss = F.cross_entropy(logits, labels)

        # Contrastive loss
        cl_loss = hierarchical_contrastive_loss(
            anchor_proj, positive_proj, negative_proj, labels
        )

        # Combined loss
        loss = ce_loss + 0.5 * cl_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_cl_loss += cl_loss.item()

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_cl_loss = total_cl_loss / len(dataloader)

    return avg_loss, avg_ce_loss, avg_cl_loss, accuracy, macro_f1


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            anchor_ids = batch['anchor_input_ids'].to(device)
            anchor_mask = batch['anchor_attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(anchor_ids, anchor_mask)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return accuracy, macro_f1, all_preds, all_labels


def train(args):
    """Main training function."""
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    vanilla_output = Path(args.vanilla_output_dir) / f"seed_{args.seed}"
    train_df = pd.read_csv(vanilla_output / "train_expanded.csv")
    val_df = pd.read_csv(vanilla_output / "val_expanded.csv")
    test_df = pd.read_csv(vanilla_output / "test_expanded.csv")

    # Load label mapping
    with open(vanilla_output / "label_mapping.json", 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    num_classes = len(label_mapping)

    # Load pretrained encoder path
    if args.pretrained_encoder_path:
        encoder_path = args.pretrained_encoder_path
    else:
        pretext_output = Path(args.pretext_output_dir) / f"seed_{args.seed}"
        encoder_path = str(pretext_output / "pretrained_encoder")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    model = HCLModel(encoder_path, num_classes)
    freeze_lower_roberta_layers(model, FROZEN_LAYER_COUNT)
    model.to(device)

    # Create datasets and dataloaders
    train_dataset = HCLDataset(train_df, tokenizer, args.max_length)
    val_dataset = HCLDataset(val_df, tokenizer, args.max_length)
    test_dataset = HCLDataset(test_df, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, ce_loss, cl_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, device, num_classes
        )
        print(f"Train Loss: {train_loss:.4f} (CE: {ce_loss:.4f}, CL: {cl_loss:.4f})")
        print(f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")

        # Evaluate
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1
            }, output_dir / "best_hcl_model.pt")

            print(f"Saved best model with validation F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "best_hcl_model.pt")['model_state_dict'])
    test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, device)

    # Generate classification report
    report = classification_report(test_labels, test_preds, target_names=list(label_mapping.values()))
    print(f"\nTest Results:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(report)

    # Save results
    results = {
        'best_val_f1': best_val_f1,
        'test_accuracy': test_acc,
        'test_macro_f1': test_f1,
        'classification_report': report
    }
    with open(output_dir / "hcl_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining completed. Best validation F1: {best_val_f1:.4f}, Test F1: {test_f1:.4f}")
    return test_f1


def main():
    parser = argparse.ArgumentParser(description="PT-HCL Stage 4: Hierarchical Contrastive Learning")
    parser.add_argument("--vanilla_output_dir", type=str, default="../1_vanilla_model/vanilla_output")
    parser.add_argument("--pretext_output_dir", type=str, default="../3_pt/pretext_output")
    parser.add_argument("--pretrained_encoder_path", type=str, default=None, help="Optional: direct path to pretrained encoder")
    parser.add_argument("--output_dir", type=str, default="./hcl_output")
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

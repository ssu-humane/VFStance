"""LKI-BART training script for textual stance classification."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup

# Label mappings
LABELS_EN = ["oppositional", "neutral", "supportive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS_EN)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# Model configuration
MODEL_CHECKPOINT = "gogamza/kobart-base-v2"
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

        # Create target with stance placeholder
        target_text = f"{STANCE_PLACEHOLDER} {ID_TO_LABEL[stance_id]}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'stance_id': stance_id
        }


class StanceHead(nn.Module):
    """Stance classification head with projection and prototype learning."""

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

    def forward(self, input_ids, attention_mask, labels=None, stance_labels=None):
        # BART forward pass
        outputs = self.bart(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

        # Get encoder hidden states
        encoder_hidden = outputs.encoder_last_hidden_state
        pooled_hidden = encoder_hidden.mean(dim=1)

        # Stance classification
        stance_logits, projected = self.stance_head(pooled_hidden)

        return outputs, stance_logits, projected


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
    accuracy = accuracy_score(gold_ids, pred_ids)
    macro_f1 = f1_score(gold_ids, pred_ids, average='macro')
    report = classification_report(gold_ids, pred_ids, output_dict=True, zero_division=0)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'f1_oppositional': report.get('0', {'f1-score': 0})['f1-score'],
        'f1_neutral': report.get('1', {'f1-score': 0})['f1-score'],
        'f1_supportive': report.get('2', {'f1-score': 0})['f1-score'],
    }


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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Add special token
    special_tokens_dict = {'additional_special_tokens': [STANCE_PLACEHOLDER]}
    tokenizer.add_special_tokens(special_tokens_dict)

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

    # Prepare datasets
    train_data = []
    val_data = []
    test_data = []

    for item in data:
        split = item.get('split')
        if split not in ('train', 'validation', 'test'):
            continue

        article_id = item.get('id')
        issue = item.get('issue', '')
        article = item.get('article', '')
        stance = item.get('stance')

        if stance is None:
            continue

        processed_item = {
            'id': article_id,
            'issue': issue,
            'article': article,
            'knowledge': knowledge_mapping.get(article_id, ''),
            'stance': stance,
            'stance_id': LABEL_TO_ID[stance]
        }

        if split == 'train':
            train_data.append(processed_item)
        elif split == 'validation':
            val_data.append(processed_item)
        elif split == 'test':
            test_data.append(processed_item)

    logger.info(f"Data loaded - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create datasets
    train_dataset = StanceDataset(train_data, tokenizer, args.max_seq_length, args.max_target_length)
    val_dataset = StanceDataset(val_data, tokenizer, args.max_seq_length, args.max_target_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = LKI_BART(MODEL_CHECKPOINT, args.projection_dim, len(LABELS_EN))
    model.bart.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0
        train_stance_loss = 0
        train_gen_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            stance_ids = batch['stance_id'].to(device)

            # Forward pass
            outputs, stance_logits, projected = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Compute losses
            gen_loss = outputs.loss
            stance_loss = F.cross_entropy(stance_logits, stance_ids)
            total_loss = args.lambda_l * gen_loss + (1 - args.lambda_l) * stance_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()
            train_gen_loss += gen_loss.item()
            train_stance_loss += stance_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_gen_loss = train_gen_loss / len(train_loader)
        avg_stance_loss = train_stance_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_golds = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                stance_ids = batch['stance_id']

                outputs, stance_logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(stance_logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_golds.extend(stance_ids.numpy())

        # Compute metrics
        val_metrics = compute_metrics(val_golds, val_preds)

        logger.info(
            f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}"
        )

        # Save best model
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            patience_counter = 0

            # Save model
            output_dir = Path(args.output_dir) / f"seed_{args.seed}"
            output_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), output_dir / "best_model.pt")
            tokenizer.save_pretrained(output_dir / "tokenizer")

            # Save config
            config = {
                'model_name': MODEL_CHECKPOINT,
                'projection_dim': args.projection_dim,
                'num_classes': len(LABELS_EN),
                'best_epoch': best_epoch,
                'best_val_f1': best_val_f1
            }
            with open(output_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved best model at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    return best_val_f1


def main():
    parser = argparse.ArgumentParser(description="Train LKI-BART for textual stance classification")
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
        default="./output",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max_target_length", type=int, default=64, help="Max target length")
    parser.add_argument("--projection_dim", type=int, default=128, help="Projection dimension")
    parser.add_argument("--lambda_l", type=float, default=0.8, help="Lambda for loss weighting")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
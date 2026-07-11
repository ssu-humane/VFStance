"""CoT-Embeddings training script for textual stance classification."""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Label mappings - converted to English for consistency
LABELS_EN = ["oppositional", "neutral", "supportive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS_EN)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# Model configuration
MODEL_CHECKPOINT = "klue/roberta-large"
FROZEN_LAYER_COUNT = 7


def freeze_lower_roberta_layers(model, frozen_layer_count):
    """Freeze the bottom encoder layers of a RoBERTa classifier."""
    base_model = getattr(model, "roberta", None)
    if base_model is None:
        return
    for layer in base_model.encoder.layer[:frozen_layer_count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_from_json(file_path):
    """Load dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text with event name and CoT."""
    sep_token = tokenizer.sep_token or "</s>"
    first_sequences = [
        f"{str(issue or '').strip()} {sep_token} {str(headline or '').strip()} {sep_token} {str(article or '').strip()}".strip()
        for issue, headline, article in zip(examples['issue'], examples['headline'], examples['text'])
    ]
    cot_sequences = [str(cot or '') for cot in examples['cot']]

    encoded = tokenizer(
        first_sequences,
        cot_sequences,
        padding='max_length',
        truncation=True,
        max_length=max_length,
    )
    return encoded


def compute_metrics(eval_pred):
    """Compute classification metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    c_report = classification_report(labels, predictions, output_dict=True, zero_division=0)

    oppositional_metrics = c_report.get('0', {'f1-score': 0, 'precision': 0, 'recall': 0})
    neutral_metrics = c_report.get('1', {'f1-score': 0, 'precision': 0, 'recall': 0})
    supportive_metrics = c_report.get('2', {'f1-score': 0, 'precision': 0, 'recall': 0})

    macro_precision = c_report['macro avg']['precision']
    macro_recall = c_report['macro avg']['recall']
    macro_f1 = c_report['macro avg']['f1-score']

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'f1_oppositional': oppositional_metrics['f1-score'],
        'f1_neutral': neutral_metrics['f1-score'],
        'f1_supportive': supportive_metrics['f1-score'],
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

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABELS_EN),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        torch_dtype=torch.float32
    )
    freeze_lower_roberta_layers(model, FROZEN_LAYER_COUNT)

    # Load data
    data = load_data_from_json(args.data_path)
    cot_data = load_data_from_json(args.cot_path) if args.cot_path else {}

    # Create CoT mapping if provided
    cot_mapping = {}
    if cot_data:
        for item in cot_data:
            cot_mapping[item.get('id')] = item.get('0_shot_cot', '')

    # Prepare datasets
    train_data = []
    validation_data = []
    test_data = []

    for item in data:
        split_value = item.get('split')
        if split_value not in ('train', 'validation', 'test'):
            continue

        article_id = item.get('id')
        headline_value = item.get('headline', '')
        text_value = item.get('article', '')
        issue_value = item.get('issue', '')
        stance = item.get('stance')

        # Get CoT if available
        cot_text = cot_mapping.get(article_id, '')

        if stance is None:
            continue

        label_id = LABEL_TO_ID[stance]

        data_item = {
            'id': article_id,
            'issue': issue_value,
            'headline': headline_value,
            'text': text_value,
            'cot': cot_text,
            'label': label_id
        }

        if split_value == 'train':
            train_data.append(data_item)
        elif split_value == 'validation':
            validation_data.append(data_item)
        elif split_value == 'test':
            test_data.append(data_item)

    logger.info(f"Data loaded - Train: {len(train_data)}, Validation: {len(validation_data)}, Test: {len(test_data)}")

    # Create datasets
    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(validation_data),
        'test': Dataset.from_list(test_data)
    })

    # Tokenize datasets
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        num_proc=4,
        desc="Running tokenizer on dataset"
    )

    # Remove unnecessary columns
    columns_to_remove = ['id', 'text', 'issue', 'headline', 'cot']
    for split in tokenized_datasets.keys():
        current_columns = tokenized_datasets[split].column_names
        remove = [col for col in columns_to_remove if col in current_columns]
        if remove:
            tokenized_datasets[split] = tokenized_datasets[split].remove_columns(remove)

    # Setup training arguments
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    logging_dir = output_dir / "logs"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy='epoch',
        save_strategy="epoch",
        save_total_limit=1,
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_dir=str(logging_dir),
        logging_strategy="epoch",
        seed=args.seed,
        fp16=True,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    # Create trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    trainer.save_model(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")

    # Evaluate on test set
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])

    # Save results
    with open(output_dir / "test_metrics.json", 'w') as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Test Results - Accuracy: {test_results.get('eval_accuracy', 0):.4f}, Macro F1: {test_results.get('eval_macro_f1', 0):.4f}")

    return test_results


def main():
    parser = argparse.ArgumentParser(description="Train CoT-Embeddings model for textual stance classification")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/k-news-stance-mm-acl-arr.json",
        help="Path to dataset JSON",
    )
    parser.add_argument(
        "--cot_path",
        type=str,
        default=None,
        help="Path to CoT dataset JSON (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Early stopping patience",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

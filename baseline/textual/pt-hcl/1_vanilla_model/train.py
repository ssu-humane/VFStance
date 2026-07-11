"""PT-HCL Stage 1: Vanilla model training with updated data format."""

import os
import json
import torch
import random
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

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
    print(f"Random seed set to {seed_value}")


def prepare_data(data_path):
    """Load and prepare data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Create label encoder for English labels
    label_encoder = LabelEncoder()
    df["stance_label"] = label_encoder.fit_transform(df["stance"])

    print("Label mapping:")
    for i, label_name in enumerate(label_encoder.classes_):
        print(f"{i}: {label_name}")

    return df, label_encoder


def freeze_lower_roberta_layers(model, frozen_layer_count):
    """Freeze the bottom encoder layers of a RoBERTa classifier."""
    base_model = getattr(model, "roberta", None)
    if base_model is None:
        return
    for layer in base_model.encoder.layer[:frozen_layer_count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def expand_dataset(df):
    """Expand dataset to sentence level for PT task."""
    expanded_data = []

    for index, row in df.iterrows():
        sentences_dict = row.get('sentences', {})
        issue = row['issue']
        headline = row.get('headline', '')
        stance_label = row['stance_label']
        split_type = row['split']
        article_id = row.get('id', index)

        # Sort sentences by index
        sentence_items = sorted(sentences_dict.items(), key=lambda item: int(item[0].split('_')[-1]))

        for i, (key, sentence) in enumerate(sentence_items):
            sentence_text = sentence.strip()
            if sentence_text:
                expanded_data.append({
                    "sentence": sentence_text,
                    "issue": issue,
                    "headline": headline,
                    "stance_label": stance_label,
                    "split": split_type,
                    "sentence_index": i,
                    "article_id": article_id
                })

    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df


def prepare_datasets(expanded_df, seed):
    """Prepare train/val/test datasets."""
    train_df = expanded_df[expanded_df["split"] == "train"].sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = expanded_df[expanded_df["split"] == "validation"].sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = expanded_df[expanded_df["split"] == "test"].sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    datasets = DatasetDict({
        'train': Dataset.from_pandas(train_df[['sentence', 'issue', 'headline', 'stance_label']]),
        'validation': Dataset.from_pandas(val_df[['sentence', 'issue', 'headline', 'stance_label']]),
        'test': Dataset.from_pandas(test_df[['sentence', 'issue', 'headline', 'stance_label']])
    })

    return datasets, train_df, val_df, test_df


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples."""
    sep_token = tokenizer.sep_token or "</s>"
    texts = [
        f"{issue} {sep_token} {headline} {sep_token} {sent}"
        for issue, headline, sent in zip(examples['issue'], examples['headline'], examples['sentence'])
    ]
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }


def train(args):
    """Main training function."""
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    df, label_encoder = prepare_data(args.data_path)
    expanded_df = expand_dataset(df)
    datasets, train_df, val_df, test_df = prepare_datasets(expanded_df, args.seed)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_encoder.classes_)
    )
    freeze_lower_roberta_layers(model, FROZEN_LAYER_COUNT)

    # Tokenize datasets
    tokenized_datasets = datasets.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        num_proc=4,
        remove_columns=['sentence', 'issue', 'headline']
    )

    # Setup training arguments
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed,
        fp16=True,
        learning_rate=args.learning_rate,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model and data
    trainer.save_model(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")

    # Save expanded dataframes for later stages
    train_df.to_csv(output_dir / "train_expanded.csv", index=False, encoding='utf-8')
    val_df.to_csv(output_dir / "val_expanded.csv", index=False, encoding='utf-8')
    test_df.to_csv(output_dir / "test_expanded.csv", index=False, encoding='utf-8')

    # Save label encoder
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(output_dir / "label_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])

    # Save test results
    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    print(f"Test Macro F1: {test_results.get('eval_macro_f1', 0):.4f}")

    return test_results


def main():
    parser = argparse.ArgumentParser(description="PT-HCL Stage 1: Vanilla Model Training")
    parser.add_argument("--data_path", type=str, default="dataset_for_vanilla.json")
    parser.add_argument("--model_name", type=str, default="klue/roberta-large")
    parser.add_argument("--output_dir", type=str, default="./vanilla_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

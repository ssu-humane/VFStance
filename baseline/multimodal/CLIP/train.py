from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel, get_linear_schedule_with_warmup

from utils import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    append_jsonl,
    build_logger,
    compute_metrics,
    default_dataset_path,
    default_image_dir,
    ensure_dir,
    load_json,
    resolve_image_file,
    save_json,
    save_jsonl,
    set_seed,
)

MODEL_NAME = "clip"
CHECKPOINT_NAME = "Bingsu/clip-vit-large-patch14-ko"


def build_processor_bundle() -> Dict[str, Any]:
    processor = AutoProcessor.from_pretrained(CHECKPOINT_NAME, use_fast=False)
    return {"processor": processor, "tokenizer": processor.tokenizer}


class CLIPStanceModel(nn.Module):
    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(CHECKPOINT_NAME)
        projection_dim = int(self.clip_model.config.projection_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_dim * 2),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        text_features = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        return self.classifier(torch.cat([text_features, image_features], dim=-1))

    def save_checkpoint(self, output_dir: str | Path) -> None:
        output_dir = ensure_dir(output_dir)
        save_file(self.state_dict(), str(Path(output_dir) / "model.safetensors"))
        save_json({"model_name": MODEL_NAME, "checkpoint": CHECKPOINT_NAME}, Path(output_dir) / "model_meta.json")

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        self.load_state_dict(load_file(str(Path(checkpoint_dir) / "model.safetensors")))


@dataclass(frozen=True)
class Example:
    id: int
    text: str
    image_path: str
    label_id: int
    gold: str


class ClipDataset(Dataset):
    def __init__(self, dataset_path: str, image_dir: str, split: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.image_dir = Path(image_dir)
        self.split = split
        self.examples = self._load_examples()

    def _build_text(self, row: Dict[str, Any]) -> str:
        issue = str(row["issue"]).strip()
        headline = str(row["headline"]).strip()
        article = str(row["article"]).strip()
        return f"Issue: {issue}. Article: {headline}, {article}"

    def _load_examples(self) -> List[Example]:
        rows = load_json(self.dataset_path)
        examples: List[Example] = []
        missing: List[str] = []
        for row in rows:
            if row.get("split") != self.split:
                continue
            image_path = resolve_image_file(self.image_dir, int(row["id"]))
            if image_path is None:
                missing.append(f'{row["id"]}_1.*')
                continue
            examples.append(Example(int(row["id"]), self._build_text(row), image_path, LABEL_TO_ID[row["stance"]], row["stance"]))
        if missing:
            raise FileNotFoundError(f"Missing images in {self.image_dir}: {missing[:5]}")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        with Image.open(example.image_path) as image:
            image = image.convert("RGB").copy()
        return {"id": example.id, "text": example.text, "image": image, "label_id": example.label_id, "gold": example.gold}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": [item["id"] for item in batch],
        "text": [item["text"] for item in batch],
        "image": [item["image"] for item in batch],
        "label_id": [item["label_id"] for item in batch],
        "gold": [item["gold"] for item in batch],
    }


@dataclass
class TrainConfig:
    dataset_path: str
    image_dir: str
    output_dir: str
    seed: int
    gpu: int
    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    text_max_length: int = 512


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())


def prepare_batch(batch: Dict[str, Any], processor_bundle, max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    encoded = processor_bundle["processor"](text=batch["text"], images=batch["image"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"].to(device, non_blocking=True),
        "attention_mask": encoded["attention_mask"].to(device, non_blocking=True),
        "pixel_values": encoded["pixel_values"].to(device, non_blocking=True),
        "labels": torch.tensor(batch["label_id"], dtype=torch.long, device=device),
    }


def evaluate(model: CLIPStanceModel, dataloader: DataLoader, processor_bundle, cfg: TrainConfig, device: torch.device, criterion: nn.Module, use_fp16: bool) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    gold_ids: List[int] = []
    pred_ids: List[int] = []
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = prepare_batch(raw_batch, processor_bundle, cfg.text_max_length, device)
            with torch.cuda.amp.autocast(enabled=use_fp16 and device.type == "cuda"):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
                loss = criterion(logits, batch["labels"])
            preds = torch.argmax(logits, dim=-1)
            total_loss += float(loss.item()) * batch["labels"].size(0)
            total_examples += batch["labels"].size(0)
            batch_gold = batch["labels"].detach().cpu().tolist()
            batch_pred = preds.detach().cpu().tolist()
            gold_ids.extend(batch_gold)
            pred_ids.extend(batch_pred)
            for article_id, pred_id, gold in zip(raw_batch["id"], batch_pred, raw_batch["gold"]):
                rows.append({"id": int(article_id), "prediction": ID_TO_LABEL[pred_id], "gold": gold, "thought": ""})
    metrics = compute_metrics(gold_ids, pred_ids)
    metrics["loss"] = float(total_loss / max(total_examples, 1))
    return {"metrics": metrics, "rows": rows}


def run_training(cfg: TrainConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    output_dir = ensure_dir(cfg.output_dir)
    best_model_dir = ensure_dir(output_dir / "best_model")
    logger = build_logger(output_dir, MODEL_NAME)
    processor_bundle = build_processor_bundle()
    train_dataset = ClipDataset(cfg.dataset_path, cfg.image_dir, "train")
    val_dataset = ClipDataset(cfg.dataset_path, cfg.image_dir, "validation")
    test_dataset = ClipDataset(cfg.dataset_path, cfg.image_dir, "test")
    save_json({"model_name": MODEL_NAME, "checkpoint": CHECKPOINT_NAME, "dataset_path": cfg.dataset_path, "image_dir": cfg.image_dir, "input_format": "Issue: {issue}. Article: {headline}, {article}", "hyperparameters": vars(cfg)}, output_dir / "config.json")
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    model = CLIPStanceModel(num_labels=len(LABEL_TO_ID)).to(device)
    train_loader = build_loader(train_dataset, cfg.batch_size, True, cfg.num_workers)
    val_loader = build_loader(val_dataset, cfg.batch_size, False, cfg.num_workers)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(math.ceil(len(train_loader) / cfg.gradient_accumulation_steps), 1) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg.warmup_ratio), total_steps)
    criterion = nn.CrossEntropyLoss()
    use_fp16 = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    best_val = float("-inf")
    patience = 0
    train_log = output_dir / "train_log.jsonl"
    if train_log.exists():
        train_log.unlink()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_gold: List[int] = []
        train_pred: List[int] = []
        train_loss = 0.0
        train_examples = 0
        for step, raw_batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False), start=1):
            batch = prepare_batch(raw_batch, processor_bundle, cfg.text_max_length, device)
            with torch.cuda.amp.autocast(enabled=use_fp16 and device.type == "cuda"):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"])
                raw_loss = criterion(logits, batch["labels"])
                loss = raw_loss / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()
            preds = torch.argmax(logits.detach(), dim=-1)
            train_gold.extend(batch["labels"].detach().cpu().tolist())
            train_pred.extend(preds.detach().cpu().tolist())
            train_loss += float(raw_loss.item()) * batch["labels"].size(0)
            train_examples += batch["labels"].size(0)
            if step % cfg.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        train_metrics = compute_metrics(train_gold, train_pred)
        val_metrics = evaluate(model, val_loader, processor_bundle, cfg, device, criterion, use_fp16)["metrics"]
        append_jsonl(train_log, {"epoch": epoch, "train_loss": float(train_loss / max(train_examples, 1)), "train_macro_f1": train_metrics["macro_f1"], "val_loss": val_metrics["loss"], "val_macro_f1": val_metrics["macro_f1"], "learning_rate": float(scheduler.get_last_lr()[0])})
        logger.info("Epoch %d | train_macro_f1=%.4f | val_macro_f1=%.4f", epoch, train_metrics["macro_f1"], val_metrics["macro_f1"])
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            patience = 0
            model.save_checkpoint(best_model_dir)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                break

    result = run_evaluation(cfg.dataset_path, cfg.image_dir, best_model_dir, output_dir / "test_eval", cfg.batch_size, cfg.num_workers, cfg.text_max_length, cfg.gpu, "test")
    save_json(result["metrics"], output_dir / "test_metrics.json")
    save_jsonl(output_dir / "results.jsonl", result["rows"])
    return result


def run_evaluation(dataset_path: str, image_dir: str, model_dir: str | Path, output_dir: str | Path, batch_size: int, num_workers: int, text_max_length: int, gpu: int, split: str) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    processor_bundle = build_processor_bundle()
    dataset = ClipDataset(dataset_path, image_dir, split)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = CLIPStanceModel(num_labels=len(LABEL_TO_ID))
    model.load_checkpoint(model_dir)
    model.to(device)
    outputs = evaluate(model, build_loader(dataset, batch_size, False, num_workers), processor_bundle, TrainConfig(dataset_path, image_dir, str(output_dir), 42, gpu, batch_size=batch_size, num_workers=num_workers, text_max_length=text_max_length), device, nn.CrossEntropyLoss(), device.type == "cuda")
    save_json(outputs["metrics"], Path(output_dir) / f"{split}_metrics.json")
    save_jsonl(Path(output_dir) / f"{split}_results.jsonl", outputs["rows"])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLIP multimodal baseline.")
    parser.add_argument("--dataset_path", type=str, default=str(default_dataset_path(__file__)))
    parser.add_argument("--image_dir", type=str, default=str(default_image_dir(__file__)))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--text_max_length", type=int, default=512)
    run_training(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()

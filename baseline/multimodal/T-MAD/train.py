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
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, ViTModel, get_linear_schedule_with_warmup

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

MODEL_NAME = "t_mad"
TEXT_ENCODER = "klue/roberta-large"
VISION_ENCODER = "google/vit-base-patch16-224"


def build_text_tokenizer():
    return AutoTokenizer.from_pretrained(TEXT_ENCODER)


def build_image_processor():
    return AutoImageProcessor.from_pretrained(VISION_ENCODER)


class ResidualReasoningBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(0.1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class TMADModel(nn.Module):
    def __init__(self, reasoning_depth: int = 5, attention_heads: int = 12, projection_dim: int = 768, share_text_target_encoder: bool = True) -> None:
        super().__init__()
        self.reasoning_depth = reasoning_depth
        self.attention_heads = attention_heads
        self.projection_dim = projection_dim
        self.share_text_target_encoder = share_text_target_encoder
        self.text_encoder = AutoModel.from_pretrained(TEXT_ENCODER)
        self.target_encoder = self.text_encoder if share_text_target_encoder else AutoModel.from_pretrained(TEXT_ENCODER)
        self.vision_encoder = ViTModel.from_pretrained(VISION_ENCODER)
        self.text_hidden_size = int(self.text_encoder.config.hidden_size)
        self.image_hidden_size = int(self.vision_encoder.config.hidden_size)
        self.text_projection = nn.Linear(self.text_hidden_size, projection_dim)
        self.target_projection = nn.Linear(self.text_hidden_size, projection_dim)
        self.image_projection = nn.Identity() if self.image_hidden_size == projection_dim else nn.Linear(self.image_hidden_size, projection_dim)
        self.text_alignment = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=attention_heads, batch_first=True)
        self.image_alignment = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=attention_heads, batch_first=True)
        self.text_weighting = nn.Sequential(nn.Linear(projection_dim * 2, projection_dim), nn.GELU(), nn.Linear(projection_dim, 1))
        self.image_weighting = nn.Sequential(nn.Linear(projection_dim * 2, projection_dim), nn.GELU(), nn.Linear(projection_dim, 1))
        reasoning_hidden = projection_dim * 3
        self.reasoning_blocks = nn.ModuleList([ResidualReasoningBlock(reasoning_hidden) for _ in range(reasoning_depth)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(reasoning_hidden),
            nn.Linear(reasoning_hidden, projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, len(LABEL_TO_ID)),
        )

    def forward(self, text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor, target_input_ids: torch.Tensor, target_attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        target_outputs = self.target_encoder(input_ids=target_input_ids, attention_mask=target_attention_mask)
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        text_hidden = self.text_projection(text_outputs.last_hidden_state)
        target_hidden = self.target_projection(target_outputs.last_hidden_state)
        image_hidden = self.image_projection(vision_outputs.last_hidden_state)
        target_query = target_hidden[:, :1, :]
        aligned_text, _ = self.text_alignment(query=target_query, key=text_hidden, value=text_hidden)
        aligned_image, _ = self.image_alignment(query=target_query, key=image_hidden, value=image_hidden)
        target_feature = target_query.squeeze(1)
        aligned_text_feature = aligned_text.squeeze(1)
        aligned_image_feature = aligned_image.squeeze(1)
        text_score = self.text_weighting(torch.cat([target_feature, aligned_text_feature], dim=-1))
        image_score = self.image_weighting(torch.cat([target_feature, aligned_image_feature], dim=-1))
        modality_weights = torch.softmax(torch.cat([text_score, image_score], dim=-1), dim=-1)
        reasoning_state = torch.cat([target_feature, modality_weights[:, 0:1] * aligned_text_feature, modality_weights[:, 1:2] * aligned_image_feature], dim=-1)
        for block in self.reasoning_blocks:
            reasoning_state = block(reasoning_state)
        return {"logits": self.classifier(reasoning_state)}

    def save_checkpoint(self, output_dir: str | Path) -> None:
        output_dir = ensure_dir(output_dir)
        state_dict = self.state_dict()
        if self.share_text_target_encoder:
            state_dict = {key: value for key, value in state_dict.items() if not key.startswith("target_encoder.")}
        save_file(state_dict, str(Path(output_dir) / "model.safetensors"))
        save_json({"model_name": MODEL_NAME, "text_encoder": TEXT_ENCODER, "vision_encoder": VISION_ENCODER, "reasoning_depth": self.reasoning_depth, "attention_heads": self.attention_heads, "projection_dim": self.projection_dim}, Path(output_dir) / "model_meta.json")

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        state_dict = load_file(str(Path(checkpoint_dir) / "model.safetensors"))
        if self.share_text_target_encoder:
            duplicated = dict(state_dict)
            for key, value in state_dict.items():
                if key.startswith("text_encoder."):
                    duplicated[key.replace("text_encoder.", "target_encoder.", 1)] = value
            state_dict = duplicated
        self.load_state_dict(state_dict)


@dataclass(frozen=True)
class Example:
    id: int
    issue: str
    text: str
    target: str
    image_path: str
    label_id: int
    gold: str


class TMADDataset(Dataset):
    def __init__(self, dataset_path: str, image_dir: str, split: str, sep_token: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.image_dir = Path(image_dir)
        self.split = split
        self.sep_token = sep_token
        self.examples = self._load_examples()

    def _build_text(self, row: Dict[str, Any]) -> str:
        issue = str(row["issue"]).strip()
        headline = str(row["headline"]).strip()
        article = str(row["article"]).strip()
        return f"{issue} {self.sep_token} {headline} {self.sep_token} {article}"

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
            issue = str(row["issue"]).strip()
            examples.append(Example(int(row["id"]), issue, self._build_text(row), issue, image_path, LABEL_TO_ID[row["stance"]], row["stance"]))
        if missing:
            raise FileNotFoundError(f"Missing images in {self.image_dir}: {missing[:5]}")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        with Image.open(example.image_path) as image:
            image = image.convert("RGB").copy()
        return {"id": example.id, "text": example.text, "target": example.target, "image": image, "label_id": example.label_id, "gold": example.gold}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": [item["id"] for item in batch],
        "text": [item["text"] for item in batch],
        "target": [item["target"] for item in batch],
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
    reasoning_depth: int = 5
    attention_heads: int = 12
    projection_dim: int = 768


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())


def prepare_batch(batch: Dict[str, Any], tokenizer, image_processor, cfg: TrainConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    text_inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=cfg.text_max_length, return_tensors="pt")
    target_inputs = tokenizer(batch["target"], padding=True, truncation=True, max_length=cfg.text_max_length, return_tensors="pt")
    image_inputs = image_processor(images=batch["image"], return_tensors="pt")
    return {
        "text_input_ids": text_inputs["input_ids"].to(device, non_blocking=True),
        "text_attention_mask": text_inputs["attention_mask"].to(device, non_blocking=True),
        "target_input_ids": target_inputs["input_ids"].to(device, non_blocking=True),
        "target_attention_mask": target_inputs["attention_mask"].to(device, non_blocking=True),
        "pixel_values": image_inputs["pixel_values"].to(device, non_blocking=True),
        "labels": torch.tensor(batch["label_id"], dtype=torch.long, device=device),
    }


def evaluate(model: TMADModel, dataloader: DataLoader, tokenizer, image_processor, cfg: TrainConfig, device: torch.device, criterion: nn.Module, use_fp16: bool) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    gold_ids: List[int] = []
    pred_ids: List[int] = []
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = prepare_batch(raw_batch, tokenizer, image_processor, cfg, device)
            with torch.cuda.amp.autocast(enabled=use_fp16 and device.type == "cuda"):
                outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
                loss = criterion(outputs["logits"], batch["labels"])
            preds = torch.argmax(outputs["logits"], dim=-1)
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
    tokenizer = build_text_tokenizer()
    image_processor = build_image_processor()
    sep_token = tokenizer.sep_token or tokenizer.eos_token
    train_dataset = TMADDataset(cfg.dataset_path, cfg.image_dir, "train", sep_token)
    val_dataset = TMADDataset(cfg.dataset_path, cfg.image_dir, "validation", sep_token)
    save_json({"model_name": MODEL_NAME, "dataset_path": cfg.dataset_path, "image_dir": cfg.image_dir, "input_format": "[CLS] target [SEP] headline [SEP] article [SEP]", "target_encoder_input": "target issue only", "hyperparameters": vars(cfg)}, output_dir / "config.json")
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    model = TMADModel(reasoning_depth=cfg.reasoning_depth, attention_heads=cfg.attention_heads, projection_dim=cfg.projection_dim, share_text_target_encoder=True).to(device)
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
            batch = prepare_batch(raw_batch, tokenizer, image_processor, cfg, device)
            with torch.cuda.amp.autocast(enabled=use_fp16 and device.type == "cuda"):
                outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
                raw_loss = criterion(outputs["logits"], batch["labels"])
                loss = raw_loss / cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()
            preds = torch.argmax(outputs["logits"].detach(), dim=-1)
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
        val_metrics = evaluate(model, val_loader, tokenizer, image_processor, cfg, device, criterion, use_fp16)["metrics"]
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

    result = run_evaluation(cfg.dataset_path, cfg.image_dir, best_model_dir, output_dir / "test_eval", cfg.batch_size, cfg.num_workers, cfg.text_max_length, cfg.reasoning_depth, cfg.attention_heads, cfg.projection_dim, cfg.gpu, "test")
    save_json(result["metrics"], output_dir / "test_metrics.json")
    save_jsonl(output_dir / "results.jsonl", result["rows"])
    return result


def run_evaluation(dataset_path: str, image_dir: str, model_dir: str | Path, output_dir: str | Path, batch_size: int, num_workers: int, text_max_length: int, reasoning_depth: int, attention_heads: int, projection_dim: int, gpu: int, split: str) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    tokenizer = build_text_tokenizer()
    image_processor = build_image_processor()
    sep_token = tokenizer.sep_token or tokenizer.eos_token
    dataset = TMADDataset(dataset_path, image_dir, split, sep_token)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = TMADModel(reasoning_depth=reasoning_depth, attention_heads=attention_heads, projection_dim=projection_dim, share_text_target_encoder=True)
    model.load_checkpoint(model_dir)
    model.to(device)
    outputs = evaluate(model, build_loader(dataset, batch_size, False, num_workers), tokenizer, image_processor, TrainConfig(dataset_path, image_dir, str(output_dir), 42, gpu, batch_size=batch_size, num_workers=num_workers, text_max_length=text_max_length, reasoning_depth=reasoning_depth, attention_heads=attention_heads, projection_dim=projection_dim), device, nn.CrossEntropyLoss(), device.type == "cuda")
    save_json(outputs["metrics"], Path(output_dir) / f"{split}_metrics.json")
    save_jsonl(Path(output_dir) / f"{split}_results.jsonl", outputs["rows"])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train T-MAD multimodal baseline.")
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
    parser.add_argument("--reasoning_depth", type=int, default=5)
    parser.add_argument("--attention_heads", type=int, default=12)
    parser.add_argument("--projection_dim", type=int, default=768)
    run_training(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()

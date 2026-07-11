from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils.data_utils import load_jsonl, load_runtime, save_json, save_jsonl, stage_output_path


LABELS = {"supportive", "neutral", "oppositional"}


def build_prompt(issue: str, text_combined: str) -> str:
    return f"Given the news text, what is the stance towards {issue}. Text: {text_combined}"


def normalize_prediction(text: str) -> str:
    candidate = (text or "").strip().lower()
    for label in LABELS:
        if label in candidate:
            return label
    return candidate


@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_length = max(feature["input_ids"].size(0) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            seq_len = feature["input_ids"].size(0)
            pad_len = max_length - seq_len
            input_ids.append(
                torch.cat(
                    [
                        feature["input_ids"],
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        feature["attention_mask"],
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                )
            )
            labels.append(
                torch.cat(
                    [
                        feature["labels"],
                        torch.full((pad_len,), -100, dtype=torch.long),
                    ]
                )
            )
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


class StanceTrainDataset(Dataset):
    def __init__(self, rows: list[dict], processor: AutoProcessor, max_length: int) -> None:
        self.rows = rows
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt = build_prompt(row["issue"], row["text_combined"])
        answer = row["stance"]

        prompt_text = f"User: {prompt}\nAssistant:"
        full_text = f"{prompt_text} {answer}"

        prompt_inputs = self.processor.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        full_inputs = self.processor.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        prompt_length = min(prompt_inputs["input_ids"].size(1), labels.size(0))
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StanceInferenceDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "id": row["id"],
            "prompt": f"User: {build_prompt(row['issue'], row['text_combined'])}\nAssistant:",
        }


def load_splits(config: dict) -> tuple[list[dict], list[dict], list[dict]]:
    output_dir = Path(config["output_dir"])
    train_rows = load_jsonl(output_dir / "train.jsonl")
    val_rows = load_jsonl(output_dir / "val.jsonl")
    test_rows = load_jsonl(output_dir / "test.jsonl")
    return train_rows, val_rows, test_rows


def resolve_model_context_length(model: torch.nn.Module, processor: AutoProcessor) -> int:
    tokenizer_max = getattr(processor.tokenizer, "model_max_length", None)
    if not isinstance(tokenizer_max, int) or tokenizer_max <= 0 or tokenizer_max > 100000:
        tokenizer_max = None

    config_candidates = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None),
        getattr(getattr(model.config, "text_config", None), "sliding_window", None),
    ]
    config_limits = [value for value in config_candidates if isinstance(value, int) and value > 0]

    if tokenizer_max is not None:
        config_limits.append(tokenizer_max)

    if not config_limits:
        raise ValueError("Unable to infer model context length.")
    return min(config_limits)


def compute_dataset_max_length(rows: list[dict], processor: AutoProcessor, model: torch.nn.Module) -> int:
    tokenizer = processor.tokenizer
    observed_max = 0
    for row in rows:
        full_text = f"User: {build_prompt(row['issue'], row['text_combined'])}\nAssistant: {row.get('stance', '')}".strip()
        token_ids = tokenizer(full_text, add_special_tokens=True, truncation=False)["input_ids"]
        observed_max = max(observed_max, len(token_ids))

    if observed_max == 0:
        raise ValueError("Dataset is empty after preprocessing.")

    model_limit = resolve_model_context_length(model, processor)
    return min(observed_max, model_limit)


def resolve_run_max_length(
    config: dict,
    processor: AutoProcessor,
    model: torch.nn.Module,
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
    model_dir: Path,
) -> int:
    configured = config["stage1"].get("max_length")
    if configured is not None:
        max_length = int(configured)
        source = "config"
    else:
        max_length = compute_dataset_max_length(train_rows + val_rows + test_rows, processor, model)
        source = "dataset_scan"

    save_json(
        model_dir / "max_length_info.json",
        {
            "max_length": max_length,
            "source": source,
            "model_context_limit": resolve_model_context_length(model, processor),
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "test_size": len(test_rows),
        },
    )
    return max_length


def build_model_and_processor(config: dict) -> tuple[torch.nn.Module, AutoProcessor]:
    model_name = config["vlm_model"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=int(config["stage1"]["lora_r"]),
        lora_alpha=int(config["stage1"]["lora_alpha"]),
        lora_dropout=float(config["stage1"]["lora_dropout"]),
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, processor


def train_model(config: dict, model: torch.nn.Module, processor: AutoProcessor, train_rows: list[dict], val_rows: list[dict], output_dir: Path, max_length: int) -> Trainer:
    train_dataset = StanceTrainDataset(train_rows, processor, max_length=max_length)
    val_dataset = StanceTrainDataset(val_rows, processor, max_length=max_length)
    tokenizer = processor.tokenizer

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        learning_rate=float(config["stage1"]["lr"]),
        num_train_epochs=int(config["stage1"]["epochs"]),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    print("[step2] Starting training")
    trainer.train()
    print("[step2] Training finished")
    print(f"[step2] Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    print("[step2] Model save finished")
    return trainer


@torch.inference_mode()
def predict_test_set(
    model: torch.nn.Module,
    processor: AutoProcessor,
    test_rows: list[dict],
    max_length: int,
) -> list[dict]:
    device = next(model.parameters()).device
    model.eval()
    tokenizer = processor.tokenizer
    predictions = []

    if hasattr(model, "generation_config") and hasattr(model.generation_config, "temperature"):
        model.generation_config.temperature = None

    print(f"[step2] Starting test inference for {len(test_rows)} examples on {device}")
    for row in test_rows:
        prompt = f"User: {build_prompt(row['issue'], row['text_combined'])}\nAssistant:"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generated = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].size(1)
        decoded = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
        predictions.append(
            {
                "id": row["id"],
                "predicted_stance": normalize_prediction(decoded),
            }
        )

    predictions.sort(key=lambda row: row["id"])
    print("[step2] Test inference finished")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-VL-7B-Instruct for text-only stance prediction.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    accelerator = Accelerator()
    config, root_dir = load_runtime(args.config)

    train_rows, val_rows, test_rows = load_splits(config)
    if not train_rows:
        raise ValueError("train.jsonl is empty.")
    if not val_rows:
        raise ValueError("val.jsonl is empty. Validation split is required for Stage 1.")

    model_dir = root_dir / "models" / "stance_predictor"
    model, processor = build_model_and_processor(config)
    max_length = resolve_run_max_length(config, processor, model, train_rows, val_rows, test_rows, model_dir)
    trainer = train_model(config, model, processor, train_rows, val_rows, model_dir, max_length)
    accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        best_checkpoint = trainer.state.best_model_checkpoint
        if best_checkpoint:
            print(f"[step2] Recording best checkpoint: {best_checkpoint}")
            save_json(
                model_dir / "best_checkpoint_info.json",
                {
                    "best_model_checkpoint": best_checkpoint,
                    "best_metric": trainer.state.best_metric,
                },
            )
        print(f"[step2] Saving processor to {model_dir}")
        processor.save_pretrained(str(model_dir))
        print("[step2] Processor save finished")

    if accelerator.is_main_process:
        inference_model = accelerator.unwrap_model(trainer.model)
        predictions = predict_test_set(inference_model, processor, test_rows, max_length)
        print(f"[step2] Writing predictions to {stage_output_path(config, 'test_stance_predictions.jsonl')}")
        save_jsonl(stage_output_path(config, "test_stance_predictions.jsonl"), predictions)
        print("[step2] Prediction save finished")
        save_json(
            model_dir / "training_summary.json",
            {
                "train_size": len(train_rows),
                "val_size": len(val_rows),
                "test_size": len(test_rows),
                "model_name": config["vlm_model"],
                "target_modules": ["q_proj", "v_proj"],
                "world_size": accelerator.num_processes,
                "max_length": max_length,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "prediction_file": str(stage_output_path(config, "test_stance_predictions.jsonl")),
            },
        )
        print(f"[step2] Training summary written to {model_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()

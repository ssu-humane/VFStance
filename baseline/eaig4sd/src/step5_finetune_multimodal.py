from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import save_and_load as peft_save_and_load
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments, set_seed

from utils.data_utils import load_jsonl, load_runtime, save_json, save_jsonl
from utils.metrics import accuracy_score, macro_f1_score


LABELS = ("supportive", "neutral", "oppositional")
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def build_prompt(issue: str, text_combined: str) -> str:
    return f"Given the news text and the image, what is the stance towards {issue}. Text: {text_combined}"


def normalize_prediction(text: str) -> str:
    candidate = (text or "").strip().lower()
    if "supportive" in candidate or "positive" in candidate or "support" in candidate or "支持" in candidate:
        return "supportive"
    if (
        "oppositional" in candidate
        or "opposed" in candidate
        or "opposition" in candidate
        or "negative" in candidate
        or "oppos" in candidate
        or "反对" in candidate
    ):
        return "oppositional"
    if "neutral" in candidate or "中立" in candidate:
        return "neutral"
    return candidate


def class_wise_f1(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label in LABELS:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores[label] = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return scores


def resolve_seeds(config: dict[str, Any]) -> list[int]:
    configured = [int(seed) for seed in config.get("stage4", {}).get("seeds", [])]
    if len(configured) >= 5:
        return configured[:5]
    return DEFAULT_SEEDS


def is_distributed_run() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def patch_peft_tensor_parallel_import() -> None:
    def _noop_maybe_shard_state_dict_for_tp(model, state_dict, adapter_name):
        return None

    peft_save_and_load._maybe_shard_state_dict_for_tp = _noop_maybe_shard_state_dict_for_tp


def run_has_saved_adapter(run_dir: Path) -> bool:
    return (run_dir / "adapter_config.json").exists() and (run_dir / "adapter_model.safetensors").exists()


def run_has_completed_summary(run_dir: Path) -> bool:
    return (run_dir / "run_summary.json").exists()


def load_run_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid run summary at {path}")
    return payload


def ensure_input_require_grads(model: torch.nn.Module) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if input_embeddings is None:
        return

    def _make_inputs_require_grad(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        output.requires_grad_(True)
        return output

    input_embeddings.register_forward_hook(_make_inputs_require_grad)


def load_split_rows(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return load_jsonl(Path(config["output_dir"]) / f"{split}.jsonl")


def resolve_image_path(root_dir: Path, split: str, row_id: int) -> Path:
    return root_dir / "output" / "images_selected" / split / f"{row_id}.png"


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


def build_messages(issue: str, text_combined: str, image_path: Path, stance: str | None = None) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": build_prompt(issue, text_combined)},
            ],
        }
    ]
    if stance is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": stance}],
            }
        )
    return messages


def compute_multimodal_max_length(
    processor: AutoProcessor,
    model: torch.nn.Module,
    root_dir: Path,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_max = 0
    longest_example: dict[str, Any] | None = None

    scan_rows: list[tuple[str, dict[str, Any], str | None]] = []
    scan_rows.extend(("train", row, row["stance"]) for row in train_rows)
    scan_rows.extend(("test", row, None) for row in test_rows)

    for split, row, stance in scan_rows:
        image_path = resolve_image_path(root_dir, split, int(row["id"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Missing selected {split} image: {image_path}")

        messages = build_messages(
            issue=row["issue"],
            text_combined=row["text_combined"],
            image_path=image_path,
            stance=stance,
        )
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=stance is None)
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            truncation=False,
            padding=False,
            return_tensors="pt",
        )
        token_length = int(model_inputs["input_ids"].size(1))
        if token_length > observed_max:
            observed_max = token_length
            longest_example = {
                "id": int(row["id"]),
                "split": split,
                "token_length": token_length,
            }

    model_limit = resolve_model_context_length(model, processor)
    return {
        "max_length": min(observed_max, model_limit),
        "observed_max": observed_max,
        "model_context_limit": model_limit,
        "longest_example": longest_example,
    }


@dataclass
class MultimodalDataCollator:
    processor: AutoProcessor
    max_length: int
    train_mode: bool

    def _tokenize_feature(self, feature: dict[str, Any]) -> dict[str, torch.Tensor]:
        prompt_messages = build_messages(
            issue=feature["issue"],
            text_combined=feature["text_combined"],
            image_path=feature["image_path"],
            stance=None,
        )
        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_images, prompt_videos = process_vision_info(prompt_messages)
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=prompt_images,
            videos=prompt_videos,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        if not self.train_mode:
            return {key: value.squeeze(0) for key, value in prompt_inputs.items()}

        full_messages = build_messages(
            issue=feature["issue"],
            text_combined=feature["text_combined"],
            image_path=feature["image_path"],
            stance=feature["stance"],
        )
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        full_images, full_videos = process_vision_info(full_messages)
        full_inputs = self.processor(
            text=[full_text],
            images=full_images,
            videos=full_videos,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        prompt_length = int(prompt_inputs["attention_mask"].sum().item())
        model_inputs = {key: value.squeeze(0) for key, value in full_inputs.items()}
        labels = model_inputs["input_ids"].clone()
        labels[: min(prompt_length, labels.size(0))] = -100
        model_inputs["labels"] = labels
        return model_inputs

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        model_features = [self._tokenize_feature(feature) for feature in features]
        max_seq_len = max(feature["input_ids"].size(0) for feature in model_features)

        batch: dict[str, list[torch.Tensor]] = {}
        for feature in model_features:
            for key, value in feature.items():
                if key in {"input_ids", "attention_mask", "labels"}:
                    if key == "input_ids":
                        pad_value = self.processor.tokenizer.pad_token_id
                    elif key == "attention_mask":
                        pad_value = 0
                    else:
                        pad_value = -100
                    pad_len = max_seq_len - value.size(0)
                    if pad_len > 0:
                        value = torch.cat([value, torch.full((pad_len,), pad_value, dtype=value.dtype)])
                batch.setdefault(key, []).append(value)

        return {key: torch.stack(values) for key, values in batch.items()}


class MultimodalTrainDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], root_dir: Path) -> None:
        self.rows = rows
        self.root_dir = root_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = resolve_image_path(self.root_dir, "train", int(row["id"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Missing selected train image: {image_path}")
        return {
            "id": int(row["id"]),
            "issue": row["issue"],
            "text_combined": row["text_combined"],
            "stance": row["stance"],
            "image_path": image_path,
        }


def build_model_and_processor(config: dict[str, Any]) -> tuple[torch.nn.Module, AutoProcessor]:
    model_name = config["vlm_model"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    ensure_input_require_grads(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=int(config["stage4"]["lora_r"]),
        lora_alpha=int(config["stage4"]["lora_alpha"]),
        lora_dropout=float(config["stage4"].get("lora_dropout", 0.05)),
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    ensure_input_require_grads(model)
    model.print_trainable_parameters()
    return model, processor


def train_one_run(
    config: dict[str, Any],
    root_dir: Path,
    seed: int,
    train_rows: list[dict[str, Any]],
    run_dir: Path,
    max_length: int,
) -> AutoProcessor:
    set_seed(seed)
    model, processor = build_model_and_processor(config)
    train_dataset = MultimodalTrainDataset(train_rows, root_dir)
    collator = MultimodalDataCollator(processor=processor, max_length=max_length, train_mode=True)
    per_device_batch_size = 1
    effective_batch_size = int(config["stage4"]["batch_size"]) * int(config["stage4"]["gradient_accumulation"])
    distributed = is_distributed_run()

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        learning_rate=float(config["stage4"]["lr"]),
        num_train_epochs=int(config["stage4"]["epochs"]),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=effective_batch_size,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        label_names=["labels"],
        eval_strategy="no",
        fsdp="full_shard auto_wrap" if distributed else "",
        fsdp_config={
            "transformer_layer_cls_to_wrap": ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"],
            "backward_prefetch": "backward_pre",
            "limit_all_gathers": True,
            "activation_checkpointing": True,
            "cpu_ram_efficient_loading": True,
            "sync_module_states": True,
            "use_orig_params": True,
        }
        if distributed
        else None,
    )
    print(
        f"[step5] seed={seed} per_device_train_batch_size={per_device_batch_size} "
        f"gradient_accumulation_steps={effective_batch_size} distributed={distributed}"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    print(f"[step5] Starting run seed={seed}")
    trainer.train()
    trainer.save_model(str(run_dir))
    processor.save_pretrained(str(run_dir))
    print(f"[step5] Finished training seed={seed}")
    return processor


def load_inference_model(config: dict[str, Any], run_dir: Path, device: str) -> tuple[torch.nn.Module, AutoProcessor]:
    model_name = config["vlm_model"]
    patch_peft_tensor_parallel_import()
    processor = AutoProcessor.from_pretrained(str(run_dir), trust_remote_code=True, use_fast=False)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(run_dir))
    model = model.to(device)
    model.eval()
    return model, processor


@torch.inference_mode()
def predict_test_rows(
    model: torch.nn.Module,
    processor: AutoProcessor,
    root_dir: Path,
    test_rows: list[dict[str, Any]],
    max_length: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    model.eval()
    tokenizer = processor.tokenizer
    predictions: list[dict[str, Any]] = []

    for row in test_rows:
        image_path = resolve_image_path(root_dir, "test", int(row["id"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Missing selected test image: {image_path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": build_prompt(row["issue"], row["text_combined"])},
                ],
            }
        ]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generated = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].size(1)
        decoded = processor.batch_decode(
            [generated[0][prompt_length:]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        predicted = normalize_prediction(decoded)
        gold = normalize_prediction(row["stance"])
        predictions.append(
            {
                "id": int(row["id"]),
                "predicted": predicted,
                "gold": gold,
                "correct": predicted == gold,
            }
        )

    predictions.sort(key=lambda row: row["id"])
    return predictions


def evaluate_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [row["gold"] for row in predictions]
    pred = [row["predicted"] for row in predictions]
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": macro_f1_score(gold, pred),
        "class_wise_f1": class_wise_f1(gold, pred),
    }


def maybe_write_training_plan(config: dict[str, Any], root_dir: Path, seeds: list[int]) -> None:
    save_json(
        root_dir / "models" / "multimodal_detector" / "training_plan.json",
        {
            "stage_config": config["stage4"],
            "selected_images": str(root_dir / "output" / "processed" / "selected_images.jsonl"),
            "runs": [
                {
                    "run_id": f"run_{seed}",
                    "seed": seed,
                    "device": config["stage4"]["device"],
                }
                for seed in seeds
            ],
            "status": "configured",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-VL-7B-Instruct with selected images for stance prediction.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config, root_dir = load_runtime(args.config)
    train_rows = load_split_rows(config, "train")
    test_rows = load_split_rows(config, "test")
    if not train_rows:
        raise ValueError("train.jsonl is empty.")
    if not test_rows:
        raise ValueError("test.jsonl is empty.")

    train_image_dir = root_dir / "output" / "images_selected" / "train"
    test_image_dir = root_dir / "output" / "images_selected" / "test"
    if not train_image_dir.exists() or not test_image_dir.exists():
        raise FileNotFoundError("Selected image directories are missing. Run step4d_pagerank_select.py first.")

    seeds = resolve_seeds(config)
    maybe_write_training_plan(config, root_dir, seeds)

    model_root = root_dir / "models" / "multimodal_detector"
    output_dir = Path(config["output_dir"])
    configured_stage4_max_length = config.get("stage4", {}).get("max_length")
    if configured_stage4_max_length is not None:
        max_length = int(configured_stage4_max_length)
        max_length_info = {
            "source": "stage4.max_length",
            "configured_max_length": configured_stage4_max_length,
            "effective_max_length": max_length,
        }
        save_json(model_root / "max_length_info.json", max_length_info)
        print(f"[step5] Using configured stage4.max_length={max_length}")
    else:
        scan_model, scan_processor = build_model_and_processor(config)
        max_length_info = compute_multimodal_max_length(
            processor=scan_processor,
            model=scan_model,
            root_dir=root_dir,
            train_rows=train_rows,
            test_rows=test_rows,
        )
        max_length = int(max_length_info["max_length"])
        max_length_info["configured_max_length"] = configured_stage4_max_length
        max_length_info["effective_max_length"] = max_length
        save_json(model_root / "max_length_info.json", max_length_info)
        print(
            f"[step5] Using multimodal max_length={max_length} "
            f"(observed_max={max_length_info['observed_max']}, model_limit={max_length_info['model_context_limit']})"
        )
        del scan_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_run: list[dict[str, Any]] = []

    for seed in seeds:
        run_dir = model_root / f"run_{seed}"
        results_path = output_dir / f"results_run_{seed}.jsonl"

        if run_has_completed_summary(run_dir):
            if is_main_process():
                print(f"[step5] Skipping seed={seed}; existing run_summary.json found")
                per_run.append(load_run_summary(run_dir / "run_summary.json"))
            distributed_barrier()
            continue

        if run_has_saved_adapter(run_dir):
            if is_main_process():
                print(f"[step5] Reusing existing adapter for seed={seed}; skipping training")
            processor = AutoProcessor.from_pretrained(str(run_dir), trust_remote_code=True, use_fast=False)
        else:
            processor = train_one_run(config, root_dir, seed, train_rows, run_dir, max_length)
        distributed_barrier()

        if is_main_process():
            inference_device = config["stage4"]["device"] if torch.cuda.is_available() else "cpu"
            model, processor = load_inference_model(config, run_dir, inference_device)
            predictions = predict_test_rows(
                model=model,
                processor=processor,
                root_dir=root_dir,
                test_rows=test_rows,
                max_length=max_length,
            )
            metrics = evaluate_predictions(predictions)
            results_path = output_dir / f"results_run_{seed}.jsonl"
            save_jsonl(results_path, predictions)
            run_summary = {
                "seed": seed,
                "run_dir": str(run_dir),
                "results_file": str(results_path),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "class_wise_f1": metrics["class_wise_f1"],
            }
            save_json(run_dir / "run_summary.json", run_summary)
            per_run.append(run_summary)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        distributed_barrier()

    if is_main_process():
        acc_values = [row["accuracy"] for row in per_run]
        f1_values = [row["macro_f1"] for row in per_run]
        final_results = {
            "mean_acc": mean(acc_values),
            "std_acc": pstdev(acc_values) if len(acc_values) > 1 else 0.0,
            "mean_f1": mean(f1_values),
            "std_f1": pstdev(f1_values) if len(f1_values) > 1 else 0.0,
            "per_run": per_run,
        }
        save_json(output_dir / "final_results.json", final_results)
        print(f"[step5] Saved final results to {output_dir / 'final_results.json'}")


if __name__ == "__main__":
    main()

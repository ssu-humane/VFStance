"""Open-weight backends.

Stages 1 and 3 run Gemma 3 12B-it through ``AutoModelForImageTextToText``; Stage 2
runs SD3.5-Medium through ``StableDiffusion3Pipeline``. Both checkpoints are gated
on the Hugging Face Hub, so run ``huggingface-cli login`` (or export ``HF_TOKEN``)
once before the first run.

Each backend holds exactly one model and frees it in ``unload()``, so a pipeline
that runs the stages in sequence never keeps two models resident.
"""

import gc

from . import BaseBackend

CLIP_TOKEN_LIMIT = 77


def _torch_dtype(name):
    import torch

    return getattr(torch, name)


def _default_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _free_cuda():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class OpenTextBackend(BaseBackend):
    """Text and multimodal generation with an open-weight LVLM (Stages 1 and 3)."""

    def __init__(
        self,
        model,
        temperature=1.0,
        max_new_tokens=1024,
        seed=42,
        dtype="bfloat16",
        device_map="auto",
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.dtype = _torch_dtype(dtype)

        print(f"[open-backend] loading {model} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model,
            torch_dtype=self.dtype,
            device_map=device_map,
        ).eval()
        torch.manual_seed(seed)
        print(f"[open-backend] {model} ready on {self.model.device}", flush=True)

    def generate(self, prompt, image=None):
        import torch

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.dtype)
        prompt_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )
        return self.processor.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        ).strip()

    def unload(self):
        self.model = None
        self.processor = None
        _free_cuda()


class OpenImageBackend(BaseBackend):
    """Text-to-image generation with an open-weight T2I model (Stage 2)."""

    def __init__(
        self,
        model,
        steps=28,
        guidance=4.5,
        t5_max_length=256,
        seed=42,
        dtype="bfloat16",
        device=None,
    ):
        import torch
        from diffusers import StableDiffusion3Pipeline

        self.model_name = model
        self.steps = steps
        self.guidance = guidance
        self.t5_max_length = t5_max_length
        self.device = device or _default_device()
        self.truncated_prompts = 0

        print(f"[open-backend] loading {model} (first run downloads the weights) ...", flush=True)
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model, torch_dtype=_torch_dtype(dtype)
        )
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        print(f"[open-backend] {model} ready on {self.device}", flush=True)

    def generate_image(self, prompt):
        clip_tokens = len(self.pipeline.tokenizer(prompt).input_ids)
        if clip_tokens > CLIP_TOKEN_LIMIT:
            # The CLIP encoders truncate at 77 tokens; T5 still sees the full prompt.
            self.truncated_prompts += 1
        return self.pipeline(
            prompt=prompt,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            max_sequence_length=self.t5_max_length,
            generator=self.generator,
        ).images[0]

    def unload(self):
        if self.truncated_prompts:
            print(
                f"[open-backend] {self.truncated_prompts} prompt(s) exceeded the "
                f"{CLIP_TOKEN_LIMIT}-token CLIP limit (T5 saw up to {self.t5_max_length} tokens)",
                flush=True,
            )
        self.pipeline = None
        _free_cuda()

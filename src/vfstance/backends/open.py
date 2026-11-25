"""Open-weight backends.

Stages 1 and 3 run an open-weight LVLM (InternVL3-14B-Instruct by default), and
Stage 2 runs SD3.5 through ``StableDiffusion3Pipeline``. Checkpoints that ship
their own modeling code are loaded with ``AutoModel`` and queried through their
``chat`` API; everything else goes through ``AutoModelForImageTextToText``.

Gated checkpoints (SD3.5, Gemma 3, ...) need ``huggingface-cli login`` or an
exported ``HF_TOKEN`` before the first run.

Each backend holds exactly one model and frees it in ``unload()``, so a pipeline
that runs the stages in sequence never keeps two models resident.
"""

import gc

from . import BaseBackend

CLIP_TOKEN_LIMIT = 77

# Vision preprocessing for chat-style checkpoints, following the InternVL model card.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TILE_SIZE = 448
MAX_TILES = 12


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


def _ships_own_modeling_code(model_name):
    """True when a checkpoint provides custom modeling code without an ITT mapping."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    auto_map = getattr(config, "auto_map", None) or {}
    return bool(auto_map) and "AutoModelForImageTextToText" not in auto_map


def _build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _split_into_tiles(image, min_num=1, max_num=MAX_TILES, image_size=TILE_SIZE, use_thumbnail=True):
    """Split an image into the aspect-ratio-matched tiles the vision tower expects."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_image = image.resize((target_width, target_height))
    tiles = []
    for index in range(blocks):
        box = (
            (index % (target_width // image_size)) * image_size,
            (index // (target_width // image_size)) * image_size,
            ((index % (target_width // image_size)) + 1) * image_size,
            ((index // (target_width // image_size)) + 1) * image_size,
        )
        tiles.append(resized_image.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


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

        self.model_name = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.dtype = _torch_dtype(dtype)
        self.chat_api = _ships_own_modeling_code(model)

        print(f"[open-backend] loading {model} ...", flush=True)
        if self.chat_api:
            from transformers import AutoModel, AutoTokenizer

            self.processor = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True, use_fast=False
            )
            self.model = AutoModel.from_pretrained(
                model,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device_map,
            ).eval()
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model,
                torch_dtype=self.dtype,
                device_map=device_map,
            ).eval()
        torch.manual_seed(seed)
        print(f"[open-backend] {model} ready on {self.model.device}", flush=True)

    def generate(self, prompt, image=None):
        if self.chat_api:
            return self._generate_chat(prompt, image)
        return self._generate_image_text_to_text(prompt, image)

    def _generation_kwargs(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
        }

    def _pixel_values(self, image):
        import torch

        tiles = _split_into_tiles(image.convert("RGB"))
        transform = _build_transform(TILE_SIZE)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        return pixel_values.to(self.dtype).to(self.model.device)

    def _generate_chat(self, prompt, image=None):
        import torch

        pixel_values = None if image is None else self._pixel_values(image)
        with torch.inference_mode():
            response = self.model.chat(
                self.processor,
                pixel_values,
                prompt,
                self._generation_kwargs(),
            )
        return str(response).strip()

    def _generate_image_text_to_text(self, prompt, image=None):
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
            outputs = self.model.generate(**inputs, **self._generation_kwargs())
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

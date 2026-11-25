"""API backends for closed models.

Gemini is the configuration reported in the paper. Other providers (GPT, Claude)
plug in by extending ``PROVIDER_PREFIXES`` and adding a ``_generate_<provider>``
method; the stage scripts only rely on ``generate`` / ``generate_image``.

API keys are read from environment variables (``GOOGLE_API_KEY`` for Gemini). As a
convenience, a ``.env`` file in the repository root is loaded if the variable is
not already set and ``python-dotenv`` is installed.
"""

import io
import os
import time

from . import BaseBackend
from ..utils import REPO_ROOT

PROVIDER_PREFIXES = (
    ("gemini", "gemini"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("claude", "anthropic"),
)

PROVIDER_KEY_ENV = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

DEFAULT_MAX_ATTEMPTS = 3
RETRY_BASE_SLEEP = 5


def resolve_provider(model):
    leaf = model.rsplit("/", 1)[-1].lower()
    for prefix, provider in PROVIDER_PREFIXES:
        if leaf.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot infer the API provider for model {model!r}. "
        f"Known prefixes: {[prefix for prefix, _ in PROVIDER_PREFIXES]}"
    )


def read_api_key(provider):
    env_name = PROVIDER_KEY_ENV[provider]
    api_key = os.getenv(env_name)
    if api_key:
        return api_key

    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None
        if load_dotenv is not None:
            load_dotenv(env_file)
            api_key = os.getenv(env_name)
    if not api_key:
        raise ValueError(
            f"{env_name} is not set. Export it, or put it in {env_file}."
        )
    return api_key


def _gemini_client(provider):
    from google import genai

    return genai.Client(api_key=read_api_key(provider))


def _call_with_retry(call, max_attempts, label):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except Exception as error:  # transient API failures: rate limits, 5xx, timeouts
            last_error = error
            if attempt == max_attempts:
                break
            sleep_seconds = RETRY_BASE_SLEEP * attempt
            print(
                f"[{label}] API call failed (attempt {attempt}/{max_attempts}): {error}. "
                f"retrying in {sleep_seconds}s",
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"{label}: API call failed after {max_attempts} attempts") from last_error


def _response_text(response):
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                return str(part_text).strip()
    return ""


def _response_images(response):
    images = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None and getattr(inline_data, "data", None):
                images.append(inline_data.data)
    return images


class ApiTextBackend(BaseBackend):
    """Text and multimodal generation with a hosted model (Stages 1 and 3)."""

    def __init__(
        self,
        model,
        temperature=1.0,
        max_output_tokens=16000,
        thinking_level="LOW",
        response_mime_type=None,
        max_attempts=DEFAULT_MAX_ATTEMPTS,
    ):
        self.model = model
        self.model_name = model
        self.provider = resolve_provider(model)
        if self.provider != "gemini":
            raise NotImplementedError(
                f"The {self.provider} provider is not wired up yet; "
                "add a branch in ApiTextBackend.generate to support it."
            )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self.response_mime_type = response_mime_type
        self.max_attempts = max_attempts
        self.client = _gemini_client(self.provider)

    def generate(self, prompt, image=None):
        from google.genai import types

        contents = [prompt] if image is None else [prompt, image]
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            response_mime_type=self.response_mime_type,
        )

        def call():
            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        response = _call_with_retry(call, self.max_attempts, "api-text")
        return _response_text(response)


class ApiImageBackend(BaseBackend):
    """Text-to-image generation with a hosted model (Stage 2)."""

    def __init__(self, model, max_attempts=DEFAULT_MAX_ATTEMPTS):
        self.model = model
        self.model_name = model
        self.provider = resolve_provider(model)
        if self.provider != "gemini":
            raise NotImplementedError(
                f"The {self.provider} provider is not wired up yet; "
                "add a branch in ApiImageBackend.generate_image to support it."
            )
        self.max_attempts = max_attempts
        self.client = _gemini_client(self.provider)

    def generate_image(self, prompt):
        from PIL import Image

        def call():
            return self.client.models.generate_content(model=self.model, contents=[prompt])

        response = _call_with_retry(call, self.max_attempts, "api-image")
        images = _response_images(response)
        if not images:
            raise RuntimeError(
                f"{self.model} returned no image for the prompt "
                "(it may have been blocked by a safety filter)"
            )
        return Image.open(io.BytesIO(images[0])).convert("RGB")

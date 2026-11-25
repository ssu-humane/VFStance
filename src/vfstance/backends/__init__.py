"""Backend factory shared by every VFStance stage.

Two interfaces are used across the pipeline:

* text / multimodal generation (Stages 1 and 3): ``generate(prompt, image=None) -> str``
* image generation (Stage 2): ``generate_image(prompt) -> PIL.Image.Image``

``api`` backends call a hosted model (closed weights); ``open`` backends load
open-weight checkpoints locally. Both expose ``unload()`` so that a stage can
release GPU memory before the next stage loads its own model, keeping the peak
footprint at one model.
"""

DEFAULT_MODELS = {
    ("api", "text"): "gemini-3-flash-preview",
    ("api", "image"): "gemini-3.1-flash-image-preview",
    ("open", "text"): "OpenGVLab/InternVL3-14B-Instruct",
    ("open", "image"): "stabilityai/stable-diffusion-3.5-large",
}

BACKENDS = ("api", "open")


class BaseBackend:
    """Common lifecycle shared by API and open-weight backends."""

    def unload(self):
        """Release any model held by this backend. API backends are a no-op."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unload()
        return False


def default_model(backend, kind):
    try:
        return DEFAULT_MODELS[(backend, kind)]
    except KeyError:
        raise ValueError(f"Unknown backend/kind combination: {backend!r}/{kind!r}") from None


def add_backend_args(parser, kind):
    """Register the ``--backend`` / ``--model`` pair used by all stage scripts."""
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="api",
        help="api: hosted closed model; open: local open-weight model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. Defaults to "
            f"{DEFAULT_MODELS[('api', kind)]} (api) or {DEFAULT_MODELS[('open', kind)]} (open)."
        ),
    )


def load_text_backend(backend, model=None, **kwargs):
    """Build a backend exposing ``generate(prompt, image=None) -> str``."""
    model = model or default_model(backend, "text")
    if backend == "api":
        from .api import ApiTextBackend

        return ApiTextBackend(model, **kwargs)
    if backend == "open":
        from .open import OpenTextBackend

        return OpenTextBackend(model, **kwargs)
    raise ValueError(f"Unknown backend: {backend!r}")


def load_text_backend_for(
    backend,
    model=None,
    temperature=1.0,
    max_tokens=None,
    seed=42,
    response_mime_type=None,
):
    """Load a text backend from stage arguments, mapping them to backend kwargs.

    ``max_tokens`` becomes ``max_output_tokens`` (api) or ``max_new_tokens`` (open);
    ``seed`` and ``response_mime_type`` only apply to the backend that supports them.
    """
    kwargs = {"temperature": temperature}
    if backend == "api":
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        kwargs["response_mime_type"] = response_mime_type
    else:
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens
        kwargs["seed"] = seed
    return load_text_backend(backend, model, **kwargs)


def load_image_backend(backend, model=None, **kwargs):
    """Build a backend exposing ``generate_image(prompt) -> PIL.Image.Image``."""
    model = model or default_model(backend, "image")
    if backend == "api":
        from .api import ApiImageBackend

        return ApiImageBackend(model, **kwargs)
    if backend == "open":
        from .open import OpenImageBackend

        return OpenImageBackend(model, **kwargs)
    raise ValueError(f"Unknown backend: {backend!r}")

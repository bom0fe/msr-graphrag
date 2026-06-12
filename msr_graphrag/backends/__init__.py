"""Backend package with lazy imports for heavy HF/vLLM dependencies."""
from .base import GenerationResult, LLMBackend, TokenAccount, simple_token_count
from .mock import MockBackend

__all__ = [
    "LLMBackend",
    "GenerationResult",
    "TokenAccount",
    "simple_token_count",
    "MockBackend",
    "HFTransformersBackend",
    "VLLMBackend",
    "build_backend",
]


def __getattr__(name):
    if name == "HFTransformersBackend":
        from .hf_transformers import HFTransformersBackend

        return HFTransformersBackend
    if name == "VLLMBackend":
        from .vllm_backend import VLLMBackend

        return VLLMBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_backend(kind: str = "mock", **kwargs) -> "LLMBackend":
    """Backend factory. kind: mock | hf | vllm."""
    kind = kind.lower()
    if kind == "mock":
        return MockBackend(**kwargs)
    if kind in ("hf", "transformers", "hf_transformers"):
        from .hf_transformers import HFTransformersBackend

        return HFTransformersBackend(**kwargs)
    if kind == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(**kwargs)
    raise ValueError(f"unknown backend kind: {kind}")

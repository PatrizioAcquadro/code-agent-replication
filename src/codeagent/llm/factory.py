"""Unified LLM factory for CodeAgent.

This module provides a single factory function to create LLM instances
from any supported provider.
"""

from typing import Tuple, Any, Optional, Literal

from .huggingface import load_huggingface_llm
from .openai_provider import create_openai_llm, create_openrouter_llm
from .gemini_provider import create_gemini_llm

# Type alias for supported LLM types
LLMType = Literal["huggingface", "openai", "gemini", "deepseek", "openrouter"]


def create_llm(
    llm_type: LLMType,
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, bool]:
    """Create an LLM instance from the specified provider.

    This factory function provides a unified interface for creating LLM
    instances from various providers.

    Args:
        llm_type: The type of LLM provider to use.
            - "huggingface": Local HuggingFace model with quantization
            - "openai": OpenAI API
            - "gemini": Google Gemini API
            - "deepseek" or "openrouter": DeepSeek via OpenRouter

        model_id: The specific model identifier. If None, uses provider defaults:
            - huggingface: "Qwen/Qwen3-4B"
            - openai: "gpt-4.1-nano-2025-04-14"
            - gemini: "gemini-2.5-flash-preview-05-20"
            - deepseek: "deepseek/deepseek-chat-v3-0324:free"

        **kwargs: Additional arguments passed to the provider-specific function.

    Returns:
        A tuple of (llm_instance, success_flag).
        For huggingface, the llm_instance is the LangChain pipeline.
        If initialization fails, returns (None, False).

    Example:
        >>> # Create OpenAI LLM
        >>> llm, ready = create_llm("openai", model_id="gpt-4")
        >>> if ready:
        ...     response = llm.invoke("Hello!")

        >>> # Create Gemini LLM
        >>> llm, ready = create_llm("gemini")
        >>> if ready:
        ...     response = llm.invoke("Hello!")

        >>> # Create HuggingFace LLM
        >>> llm, ready = create_llm("huggingface", model_id="Qwen/Qwen3-4B")
    """
    llm_type = llm_type.lower()

    if llm_type == "huggingface":
        model_id = model_id or "Qwen/Qwen3-4B"
        _, _, llm, success = load_huggingface_llm(model_id, **kwargs)
        return llm, success

    elif llm_type == "openai":
        model_id = model_id or "gpt-4.1-nano-2025-04-14"
        return create_openai_llm(model=model_id, **kwargs)

    elif llm_type == "gemini":
        model_id = model_id or "gemini-2.5-flash-preview-05-20"
        return create_gemini_llm(model=model_id, **kwargs)

    elif llm_type in ("deepseek", "openrouter"):
        model_id = model_id or "deepseek/deepseek-chat-v3-0324:free"
        return create_openrouter_llm(model=model_id, **kwargs)

    else:
        print(f"Unknown LLM type: '{llm_type}'")
        print("Supported types: huggingface, openai, gemini, deepseek, openrouter")
        return None, False


def get_available_llms() -> dict[str, Tuple[Any, bool]]:
    """Attempt to initialize all available LLMs.

    Returns:
        A dictionary mapping LLM names to (instance, ready) tuples.
    """
    available = {}

    for llm_type in ["openai", "gemini", "deepseek"]:
        try:
            llm, ready = create_llm(llm_type)
            available[llm_type] = (llm, ready)
        except Exception:
            available[llm_type] = (None, False)

    return available

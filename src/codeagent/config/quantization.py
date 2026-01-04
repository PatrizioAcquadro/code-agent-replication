"""Quantization configuration for HuggingFace models.

This module provides factory functions for creating quantization configurations,
particularly for 4-bit quantization using BitsAndBytes.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig


def create_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> "BitsAndBytesConfig":
    """Create a BitsAndBytes quantization configuration.

    This factory function creates a standard 4-bit quantization configuration
    optimized for running large language models in resource-constrained environments.

    Args:
        load_in_4bit: Whether to load the model in 4-bit precision.
        bnb_4bit_quant_type: The quantization type ("nf4" or "fp4").
        bnb_4bit_use_double_quant: Whether to use nested quantization.

    Returns:
        A configured BitsAndBytesConfig instance.

    Example:
        >>> config = create_bnb_config()
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     model_id,
        ...     quantization_config=config,
        ...     device_map="auto"
        ... )
    """
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


# Default quantization config (lazy-loaded)
_default_bnb_config = None


def get_default_bnb_config() -> "BitsAndBytesConfig":
    """Get the default BitsAndBytes configuration.

    This function lazily creates and caches the default configuration.

    Returns:
        The default BitsAndBytesConfig instance.
    """
    global _default_bnb_config
    if _default_bnb_config is None:
        _default_bnb_config = create_bnb_config()
    return _default_bnb_config

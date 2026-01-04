"""HuggingFace model loading with quantization support.

This module provides functions to load HuggingFace models with 4-bit
quantization for efficient inference.
"""

from typing import Tuple, Any, Optional, TYPE_CHECKING

from ..config.settings import get_generation_config

if TYPE_CHECKING:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from langchain_huggingface import HuggingFacePipeline


def load_huggingface_llm(
    model_id: str,
    bnb_config: Optional["BitsAndBytesConfig"] = None,
    max_new_tokens: int = 1024,
) -> Tuple[Any, Any, Any, bool]:
    """Load a HuggingFace model with quantization for LangChain integration.

    This function loads a specified HuggingFace model and tokenizer with
    4-bit quantization, wrapping them in a LangChain-compatible pipeline.

    Args:
        model_id: The HuggingFace model identifier (e.g., "Qwen/Qwen3-4B").
        bnb_config: BitsAndBytes quantization configuration. If None, creates default.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        A tuple of (model, tokenizer, langchain_pipeline, success_flag).
        If loading fails, returns (None, None, None, False).

    Example:
        >>> from codeagent.config import create_bnb_config
        >>> bnb_config = create_bnb_config()
        >>> model, tokenizer, llm, success = load_huggingface_llm(
        ...     "Qwen/Qwen3-4B", bnb_config
        ... )
        >>> if success:
        ...     response = llm.invoke("Hello, world!")
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFacePipeline

    print(f"--- Attempting to load model: {model_id} ---")

    # Get model-specific generation config
    gen_config = get_generation_config(model_id)

    try:
        # Create default quantization config if not provided
        if bnb_config is None:
            from ..config.quantization import create_bnb_config
            bnb_config = create_bnb_config()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Ensure pad_token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer pad_token set to eos_token: {tokenizer.eos_token}")

        # Load the quantized model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Create the LangChain pipeline
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            max_new_tokens=max_new_tokens,
            **gen_config,
        )
        llm_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)

        print(f"--- Successfully loaded and wrapped model: {model_id} ---\n")
        return model, tokenizer, llm_pipeline, True

    except Exception as e:
        print(f"\nERROR: Failed to load model or tokenizer for '{model_id}'.")
        print(f"Details: {e}\n")
        return None, None, None, False


def build_chat_prompt(
    tokenizer: Any,
    system_message: str,
    user_message: str,
    enable_thinking: bool = False,
) -> Optional[str]:
    """Build a formatted chat prompt using the tokenizer's template.

    Args:
        tokenizer: The HuggingFace tokenizer.
        system_message: The system prompt content.
        user_message: The user's message content.
        enable_thinking: Whether to enable thinking mode for supported models.

    Returns:
        The formatted prompt string, or None if formatting fails.
    """
    if not tokenizer:
        print("ERROR: Tokenizer is not available.")
        return None

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    try:
        prompt_string = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return prompt_string
    except Exception as e:
        print(f"ERROR: Failed during prompt construction: {e}")
        return None

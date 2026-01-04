"""OpenAI and OpenRouter LLM provider integration.

This module provides functions to create OpenAI and OpenRouter-compatible
LLM instances for use with LangChain.
"""

import os
from typing import Tuple, Optional, Any

from ..config.secrets import get_secret


def create_openai_llm(
    model: str = "gpt-4.1-nano-2025-04-14",
    api_key: Optional[str] = None,
) -> Tuple[Any, bool]:
    """Create an OpenAI ChatOpenAI instance.

    Args:
        model: The OpenAI model name.
        api_key: API key. If None, reads from OPENAI_API_KEY environment variable.

    Returns:
        A tuple of (ChatOpenAI instance, success_flag).
        If initialization fails, returns (None, False).

    Example:
        >>> llm, ready = create_openai_llm("gpt-4")
        >>> if ready:
        ...     response = llm.invoke("Hello!")
    """
    from langchain_openai import ChatOpenAI

    # Get API key
    if api_key is None:
        api_key = get_secret("OPENAI_API_KEY")

    if not api_key:
        print("OpenAI API key not found - OpenAI disabled.")
        return None, False

    # Ensure key is in environment for LangChain
    os.environ["OPENAI_API_KEY"] = api_key

    try:
        llm = ChatOpenAI(model=model)
        print(f"OpenAI model '{model}' initialized.")
        return llm, True
    except Exception as e:
        print(f"OpenAI initialization failed: {e}")
        return None, False


def create_openrouter_llm(
    model: str = "deepseek/deepseek-chat-v3-0324:free",
    api_key: Optional[str] = None,
    referer: str = "https://github.com/PatrizioAcquadro/code-agent-replication",
    title: str = "CodeAgent Replication",
) -> Tuple[Any, bool]:
    """Create an OpenRouter-compatible ChatOpenAI instance.

    OpenRouter provides access to various LLMs including DeepSeek through
    an OpenAI-compatible API.

    Args:
        model: The model name on OpenRouter.
        api_key: API key. If None, reads from OPENROUTER_API_KEY environment variable.
        referer: HTTP Referer header for OpenRouter.
        title: X-Title header for OpenRouter.

    Returns:
        A tuple of (ChatOpenAI instance, success_flag).
        If initialization fails, returns (None, False).

    Example:
        >>> llm, ready = create_openrouter_llm("deepseek/deepseek-chat")
        >>> if ready:
        ...     response = llm.invoke("Hello!")
    """
    from langchain_openai import ChatOpenAI

    # Get API key
    if api_key is None:
        api_key = get_secret("OPENROUTER_API_KEY")

    if not api_key:
        print("OpenRouter API key not found - DeepSeek disabled.")
        return None, False

    try:
        llm = ChatOpenAI(
            model_name=model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=api_key,
            default_headers={
                "HTTP-Referer": referer,
                "X-Title": title,
            },
        )
        print(f"OpenRouter model '{model}' initialized (DeepSeek).")
        return llm, True
    except Exception as e:
        print(f"OpenRouter/DeepSeek initialization failed: {e}")
        return None, False

"""Google Gemini LLM provider integration.

This module provides functions to create Google Gemini LLM instances
for use with LangChain.
"""

import os
from typing import Tuple, Optional, Any

from ..config.secrets import get_secret


def create_gemini_llm(
    model: str = "gemini-2.5-flash-preview-05-20",
    api_key: Optional[str] = None,
) -> Tuple[Any, bool]:
    """Create a Google Gemini ChatGoogleGenerativeAI instance.

    Args:
        model: The Gemini model name.
        api_key: API key. If None, reads from GOOGLE_API_KEY environment variable.

    Returns:
        A tuple of (ChatGoogleGenerativeAI instance, success_flag).
        If initialization fails, returns (None, False).

    Example:
        >>> llm, ready = create_gemini_llm("gemini-2.5-flash")
        >>> if ready:
        ...     response = llm.invoke("Hello!")
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Get API key
    if api_key is None:
        api_key = get_secret("GOOGLE_API_KEY")

    if not api_key:
        print("Google API key not found - Gemini disabled.")
        return None, False

    # Ensure key is in environment for LangChain
    os.environ["GOOGLE_API_KEY"] = api_key

    try:
        llm = ChatGoogleGenerativeAI(model=model)
        print(f"Gemini model '{model}' initialized.")
        return llm, True
    except Exception as e:
        print(f"Gemini initialization failed: {e}")
        return None, False

"""LLM integration module for CodeAgent.

This module provides unified interfaces for loading and configuring
various LLM providers including:
- HuggingFace models with quantization
- OpenAI/OpenRouter API
- Google Gemini API
"""

from .huggingface import load_huggingface_llm
from .openai_provider import create_openai_llm, create_openrouter_llm
from .gemini_provider import create_gemini_llm
from .factory import create_llm, LLMType

__all__ = [
    "load_huggingface_llm",
    "create_openai_llm",
    "create_openrouter_llm",
    "create_gemini_llm",
    "create_llm",
    "LLMType",
]

"""Configuration module for CodeAgent.

This module provides centralized configuration management, including:
- Project settings and paths
- LLM configuration
- Evaluation settings
- Secret/API key management
- Quantization configuration for HuggingFace models
"""

from .settings import (
    ProjectConfig,
    LLMConfig,
    EvaluationConfig,
    GENERATION_CONFIGS,
    get_generation_config,
)
from .secrets import get_secret
from .quantization import create_bnb_config

__all__ = [
    "ProjectConfig",
    "LLMConfig",
    "EvaluationConfig",
    "GENERATION_CONFIGS",
    "get_generation_config",
    "get_secret",
    "create_bnb_config",
]

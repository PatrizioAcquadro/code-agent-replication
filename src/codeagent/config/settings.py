"""Configuration dataclasses for CodeAgent.

This module defines immutable configuration objects for various aspects of the framework:
- ProjectConfig: Paths and project-level settings
- LLMConfig: Language model configuration
- EvaluationConfig: Evaluation and testing parameters
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import os


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable project configuration.

    Attributes:
        project_repo_path: Base path for the working repository.
        grammar_path: Path for tree-sitter grammar files.
        random_seed: Seed for reproducibility.
    """
    project_repo_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CODEAGENT_REPO_PATH", "./mini_transformers_repo")
        )
    )
    grammar_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CODEAGENT_GRAMMAR_PATH", "./grammars")
        )
    )
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        # Convert string paths to Path objects if needed
        object.__setattr__(self, 'project_repo_path', Path(self.project_repo_path))
        object.__setattr__(self, 'grammar_path', Path(self.grammar_path))


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM providers.

    Attributes:
        default_temperature: Default sampling temperature.
        default_top_p: Default nucleus sampling parameter.
        default_top_k: Default top-k sampling parameter.
        max_new_tokens: Maximum tokens to generate.
        timeout_seconds: API call timeout.
    """
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 50
    max_new_tokens: int = 1024
    timeout_seconds: int = 60
    do_sample: bool = True

    def to_generation_kwargs(self) -> Dict[str, Any]:
        """Convert to generation keyword arguments.

        Returns:
            Dictionary of generation parameters.
        """
        return {
            "temperature": self.default_temperature,
            "top_p": self.default_top_p,
            "top_k": self.default_top_k,
            "do_sample": self.do_sample,
        }


# Model-specific generation configurations
GENERATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "do_sample": True,
    },
    "Qwen/Qwen3-4B": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "do_sample": True,
    },
    "Qwen/Qwen3-8B": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "do_sample": True,
    },
    "codellama/CodeLlama-7b-Instruct-hf": {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": True,
    },
}


def get_generation_config(model_id: str) -> Dict[str, Any]:
    """Get generation configuration for a specific model.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        Dictionary of generation parameters for the model.
    """
    return GENERATION_CONFIGS.get(model_id, GENERATION_CONFIGS["default"])


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluation runs.

    Attributes:
        max_agent_iterations: Maximum steps for agent execution.
        code_execution_timeout: Timeout for code execution in seconds.
        pause_between_tasks: Pause duration between evaluation tasks.
        test_timeout: Timeout for running tests.
    """
    max_agent_iterations: int = 25
    code_execution_timeout: int = 20
    pause_between_tasks: int = 15
    test_timeout: int = 60


# Default configuration instances
DEFAULT_PROJECT_CONFIG = ProjectConfig()
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()

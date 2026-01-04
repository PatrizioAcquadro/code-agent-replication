"""CodeAgent: LLM-based Agent Framework for Repository-level Code Generation.

CodeAgent is a replication of the framework described in https://arxiv.org/abs/2401.07339.
It leverages external tools to assist LLMs in repository-level code generation tasks,
where generating code requires understanding all software artifacts in a repository
(documentation, code dependencies, runtime environment).

Main Components:
- config: Configuration settings, API keys, and quantization
- llm: LLM providers (HuggingFace, OpenAI, Gemini, DeepSeek)
- tools: Programming tools (FormatCheck, CodeNavigation, Interpreter, DocSearch, WebSearch)
- benchmarks: Benchmark datasets (CodeAgentBench, MiniTransformers)
- agents: Agent factory and strategies
- evaluation: Evaluation pipeline and metrics
- utils: Utility functions

Quick Start:
    >>> from codeagent import CodeAgentConfig, create_llm, get_all_tools
    >>> from codeagent.agents import create_agent_executor
    >>> from codeagent.benchmarks import MiniTransformersBench
    >>> from codeagent.evaluation import run_evaluation_pipeline
    >>>
    >>> # Configuration
    >>> config = CodeAgentConfig()
    >>>
    >>> # Load LLM
    >>> llm, ready = create_llm("gemini")
    >>>
    >>> # Setup tools
    >>> tools = get_all_tools(config.project_repo_path)
    >>>
    >>> # Create agent
    >>> agent = create_agent_executor(llm, tools, strategy="react")
    >>>
    >>> # Run evaluation
    >>> benchmark = MiniTransformersBench()
    >>> results = run_evaluation_pipeline(agent, benchmark)
"""

__version__ = "0.1.0"

# Configuration exports
from .config import (
    ProjectConfig,
    LLMConfig,
    EvaluationConfig,
    GENERATION_CONFIGS,
    get_secret,
    create_bnb_config,
)

# Convenience alias for project configuration
CodeAgentConfig = ProjectConfig

# LLM factory
from .llm import create_llm

# Tools factory
from .tools import get_all_tools

# Utility functions
from .utils import (
    fix_random_seeds,
    safe_join_path,
    read_file_content,
    write_file_content,
    list_files_in_repo,
    clean_final_code,
    extract_code_from_markdown,
    generate_api_guide,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "CodeAgentConfig",
    "ProjectConfig",
    "LLMConfig",
    "EvaluationConfig",
    "GENERATION_CONFIGS",
    "get_secret",
    "create_bnb_config",
    # LLM
    "create_llm",
    # Tools
    "get_all_tools",
    # Utilities
    "fix_random_seeds",
    "safe_join_path",
    "read_file_content",
    "write_file_content",
    "list_files_in_repo",
    "clean_final_code",
    "extract_code_from_markdown",
    "generate_api_guide",
]

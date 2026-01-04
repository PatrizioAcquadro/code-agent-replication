"""Benchmark module for CodeAgent.

This module provides benchmark datasets for evaluating the agent:
- CodeAgentBench: 57 tasks from numpy-ml repository
- MiniTransformersBench: 15 tasks for iterative development
"""

from .base import Benchmark, BenchmarkTask
from .loader import load_jsonl, load_benchmark_from_url
from .analysis import (
    force_string,
    extract_imports,
    analyze_code_symbols,
    get_task_type,
    get_target_path_from_annotation,
)
from .codeagent_bench import CodeAgentBench
from .mini_transformers import MiniTransformersBench

__all__ = [
    "Benchmark",
    "BenchmarkTask",
    "load_jsonl",
    "load_benchmark_from_url",
    "force_string",
    "extract_imports",
    "analyze_code_symbols",
    "get_task_type",
    "get_target_path_from_annotation",
    "CodeAgentBench",
    "MiniTransformersBench",
]

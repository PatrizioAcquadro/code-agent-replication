"""Utility functions for CodeAgent.

This module provides shared utilities for:
- Safe file operations within project boundaries
- Random seed management for reproducibility
- Code extraction and cleaning from LLM outputs
- API documentation generation
"""

from .file_ops import (
    safe_join_path,
    read_file_content,
    write_file_content,
    list_files_in_repo,
)
from .seed import fix_random_seeds
from .code_cleaning import clean_final_code, extract_code_from_markdown
from .api_guide import generate_api_guide

__all__ = [
    "safe_join_path",
    "read_file_content",
    "write_file_content",
    "list_files_in_repo",
    "fix_random_seeds",
    "clean_final_code",
    "extract_code_from_markdown",
    "generate_api_guide",
]

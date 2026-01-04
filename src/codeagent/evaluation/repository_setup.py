"""Repository setup utilities for evaluation.

This module provides functions to prepare the repository for each
evaluation task, ensuring a clean state.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tree_sitter import Parser


def setup_repository_for_task(
    task_id: str,
    codebase_df: pd.DataFrame,
    task_df: pd.DataFrame,
    project_repo_path: Path,
    parser: Optional["Parser"] = None,
) -> None:
    """Reset and prepare the repository for a specific evaluation task.

    This function:
    1. Clears the existing repository
    2. Reconstructs all files from the codebase
    3. For CREATE tasks, removes the target file
    4. Generates API documentation if a parser is available

    Args:
        task_id: The ID of the task to set up for.
        codebase_df: DataFrame containing the codebase files.
        task_df: DataFrame containing task definitions.
        project_repo_path: Path to the project repository.
        parser: Optional tree-sitter parser for API guide generation.

    Example:
        >>> setup_repository_for_task(
        ...     "miniformer-01",
        ...     codebase_df,
        ...     task_df,
        ...     Path("./mini_transformers_repo"),
        ... )
    """
    from ..utils.file_ops import safe_join_path, write_file_content
    from ..utils.api_guide import generate_api_guide

    print(f"\n--- Setting up repository for Task ID: {task_id} ---")

    project_repo_path = Path(project_repo_path)

    # Clear and recreate the repository
    if project_repo_path.exists():
        shutil.rmtree(project_repo_path)
    project_repo_path.mkdir(parents=True, exist_ok=True)

    # Reconstruct all files from the base codebase
    for _, row in codebase_df.iterrows():
        content = row["content"]
        if isinstance(content, list):
            content = "".join(content)
        write_file_content(row["path"], content, project_repo_path)

    # Get task info
    task_info = task_df[task_df["task_id"] == task_id].iloc[0]
    target_file = task_info["class_link"]

    # For CREATE tasks, remove the target file
    create_tasks = ["PositionalEmbedding", "Miniformer", "LanguageModelHead"]
    if task_info["title"] in create_tasks:
        try:
            full_path = safe_join_path(project_repo_path, target_file)
            if os.path.exists(full_path):
                os.remove(full_path)
                print(f"Task is CREATE. Removed existing file: {target_file}")
        except Exception as e:
            print(f"Warning: Could not remove file for CREATE task: {e}")

    # Generate API documentation
    if parser is not None:
        try:
            api_guide_md = generate_api_guide(project_repo_path, parser)
            write_file_content("project_docs/api_guide.md", api_guide_md, project_repo_path)
            print("API documentation generated.")
        except Exception as e:
            print(f"Warning: Could not generate API guide: {e}")
    else:
        print("Warning: Tree-sitter parser not available. API guide not generated.")

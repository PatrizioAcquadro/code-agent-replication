"""Base classes for benchmarks.

This module defines the abstract base class for benchmarks and
common data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import pandas as pd


@dataclass
class BenchmarkTask:
    """Represents a single benchmark task.

    Attributes:
        task_id: Unique identifier for the task.
        title: Short title or name of the task.
        comment: The full instruction/prompt for the task.
        class_link: Path to the target file.
        class_name: Name of the class to generate/modify.
        class_annotation: Full annotation path (e.g., 'module.ClassName').
        test_file_path: Path to the test file for verification.
    """

    task_id: str
    title: str
    comment: str
    class_link: str
    class_name: str
    class_annotation: str
    test_file_path: str

    @classmethod
    def from_series(cls, row: pd.Series) -> "BenchmarkTask":
        """Create a BenchmarkTask from a pandas Series.

        Args:
            row: A pandas Series containing task data.

        Returns:
            A BenchmarkTask instance.
        """
        return cls(
            task_id=row.get("task_id", ""),
            title=row.get("title", ""),
            comment=row.get("comment", ""),
            class_link=row.get("class_link", ""),
            class_name=row.get("class_name", ""),
            class_annotation=row.get("class_annotation", ""),
            test_file_path=row.get("test_file_path", ""),
        )


class Benchmark(ABC):
    """Abstract base class for code generation benchmarks.

    Subclasses must implement methods to load the codebase and tasks,
    and can optionally override reconstruction logic.
    """

    def __init__(self, name: str) -> None:
        """Initialize the benchmark.

        Args:
            name: The name of the benchmark.
        """
        self.name = name
        self._codebase_df: Optional[pd.DataFrame] = None
        self._tasks_df: Optional[pd.DataFrame] = None

    @abstractmethod
    def load_codebase(self) -> pd.DataFrame:
        """Load the codebase files.

        Returns:
            A DataFrame with 'path' and 'content' columns.
        """
        pass

    @abstractmethod
    def load_tasks(self) -> pd.DataFrame:
        """Load the benchmark tasks.

        Returns:
            A DataFrame with task definitions.
        """
        pass

    @property
    def codebase(self) -> pd.DataFrame:
        """Get the codebase DataFrame, loading if necessary."""
        if self._codebase_df is None:
            self._codebase_df = self.load_codebase()
        return self._codebase_df

    @property
    def tasks(self) -> pd.DataFrame:
        """Get the tasks DataFrame, loading if necessary."""
        if self._tasks_df is None:
            self._tasks_df = self.load_tasks()
        return self._tasks_df

    def get_task(self, task_id: str) -> BenchmarkTask:
        """Get a specific task by ID.

        Args:
            task_id: The task identifier.

        Returns:
            A BenchmarkTask instance.

        Raises:
            KeyError: If the task is not found.
        """
        tasks_df = self.tasks
        task_row = tasks_df[tasks_df["task_id"] == task_id]
        if task_row.empty:
            raise KeyError(f"Task '{task_id}' not found in benchmark.")
        return BenchmarkTask.from_series(task_row.iloc[0])

    def get_task_ids(self) -> list[str]:
        """Get all task IDs in the benchmark.

        Returns:
            List of task ID strings.
        """
        return self.tasks["task_id"].tolist()

    def reconstruct_repository(
        self,
        target_path: Path,
        parser: Optional[Any] = None,
    ) -> None:
        """Reconstruct the repository on the filesystem.

        Args:
            target_path: The path where the repository should be created.
            parser: Optional tree-sitter parser for generating API docs.
        """
        import os
        import shutil

        from ..utils.file_ops import write_file_content
        from ..utils.api_guide import generate_api_guide

        # Clean and recreate target directory
        if target_path.exists():
            shutil.rmtree(target_path)
        target_path.mkdir(parents=True, exist_ok=True)

        # Write all files from codebase
        codebase = self.codebase
        for _, row in codebase.iterrows():
            content = row["content"]
            if isinstance(content, list):
                content = "".join(content)
            write_file_content(row["path"], content, target_path)

        print(f"Repository reconstructed at: {target_path}")

        # Generate API documentation if parser is available
        if parser is not None:
            try:
                api_guide = generate_api_guide(target_path, parser)
                write_file_content("project_docs/api_guide.md", api_guide, target_path)
                print("API guide generated.")
            except Exception as e:
                print(f"Warning: Could not generate API guide: {e}")

"""CodeAgentBench: Main benchmark with 57 tasks from numpy-ml.

This benchmark contains tasks for generating classes and functions
from the numpy-ml repository.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import Benchmark
from .loader import load_benchmark_from_url


class CodeAgentBench(Benchmark):
    """The main CodeAgentBench with 57 tasks from numpy-ml.

    This benchmark tests repository-level code generation capabilities
    using tasks that require understanding of the full numpy-ml codebase.

    Attributes:
        CODEBASE_URL: URL to the codebase JSONL file.
        TASKS_URL: URL to the tasks JSONL file.
    """

    CODEBASE_URL = (
        "https://raw.githubusercontent.com/PatrizioAcquadro/"
        "code-agent-replication/refs/heads/main/datasets/all_py_content.jsonl"
    )
    TASKS_URL = (
        "https://raw.githubusercontent.com/PatrizioAcquadro/"
        "code-agent-replication/refs/heads/main/datasets/final_dataset.jsonl"
    )

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize the benchmark.

        Args:
            cache_dir: Optional directory to cache downloaded files.
        """
        super().__init__("CodeAgentBench")
        self.cache_dir = cache_dir

    def load_codebase(self) -> pd.DataFrame:
        """Load the numpy-ml codebase.

        Returns:
            A DataFrame with 'path' and 'content' columns.
        """
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / "all_py_content.jsonl"

        df = load_benchmark_from_url(self.CODEBASE_URL, cache_path)
        print(f"Loaded {len(df)} files from CodeAgentBench codebase.")
        return df

    def load_tasks(self) -> pd.DataFrame:
        """Load the 57 benchmark tasks.

        Returns:
            A DataFrame with task definitions.
        """
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / "final_dataset.jsonl"

        df = load_benchmark_from_url(self.TASKS_URL, cache_path)
        print(f"Loaded {len(df)} tasks from CodeAgentBench.")
        return df

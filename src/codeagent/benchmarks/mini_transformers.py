"""MiniTransformersBench: Smaller benchmark for iterative development.

This benchmark contains 15 tasks for faster iteration during development.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import Benchmark
from .loader import load_benchmark_from_url


class MiniTransformersBench(Benchmark):
    """A smaller benchmark with 15 tasks for iterative development.

    This benchmark uses a simpler miniformer codebase, making it
    faster to test and iterate on agent improvements.

    Attributes:
        CODEBASE_URL: URL to the codebase JSONL file.
        TASKS_URL: URL to the tasks JSONL file.
    """

    CODEBASE_URL = (
        "https://raw.githubusercontent.com/PatrizioAcquadro/"
        "code-agent-replication/refs/heads/main/datasets/"
        "mini_transformers_codebase.jsonl"
    )
    TASKS_URL = (
        "https://raw.githubusercontent.com/PatrizioAcquadro/"
        "code-agent-replication/refs/heads/main/datasets/"
        "mini_transformers_tasks.jsonl"
    )

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize the benchmark.

        Args:
            cache_dir: Optional directory to cache downloaded files.
        """
        super().__init__("MiniTransformersBench")
        self.cache_dir = cache_dir

    def load_codebase(self) -> pd.DataFrame:
        """Load the miniformer codebase.

        Returns:
            A DataFrame with 'path' and 'content' columns.
        """
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / "mini_transformers_codebase.jsonl"

        df = load_benchmark_from_url(self.CODEBASE_URL, cache_path)
        print(f"Loaded {len(df)} files from MiniTransformersBench codebase.")
        return df

    def load_tasks(self) -> pd.DataFrame:
        """Load the 15 benchmark tasks.

        Returns:
            A DataFrame with task definitions.
        """
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / "mini_transformers_tasks.jsonl"

        df = load_benchmark_from_url(self.TASKS_URL, cache_path)
        print(f"Loaded {len(df)} tasks from MiniTransformersBench.")
        return df

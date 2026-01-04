"""JSONL loading utilities for benchmarks.

This module provides functions to load benchmark data from JSONL files,
either from local paths or URLs.
"""

from pathlib import Path
from typing import Union, Optional

import pandas as pd


def load_jsonl(path: Union[str, Path]) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame.

    Args:
        path: Path to the JSONL file.

    Returns:
        A DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    try:
        return pd.read_json(path, lines=True)
    except Exception as e:
        raise ValueError(f"Failed to parse JSONL file {path}: {e}")


def load_benchmark_from_url(
    url: str,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load a benchmark JSONL file from a URL.

    Optionally caches the downloaded file for future use.

    Args:
        url: The URL to the JSONL file.
        cache_path: Optional local path to cache the file.

    Returns:
        A DataFrame with the loaded data.

    Raises:
        RuntimeError: If the download or parsing fails.
    """
    # Check cache first
    if cache_path and cache_path.exists():
        try:
            return pd.read_json(cache_path, lines=True)
        except Exception:
            pass  # Cache invalid, redownload

    try:
        df = pd.read_json(url, lines=True)

        # Cache if path provided
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_json(cache_path, orient="records", lines=True)

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to load benchmark from {url}: {e}")

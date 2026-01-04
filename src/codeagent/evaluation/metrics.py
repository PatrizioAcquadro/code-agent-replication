"""Evaluation metrics and reporting.

This module provides functions for calculating metrics and
generating reports from evaluation results.
"""

from typing import List, Union

import pandas as pd

from .task_runner import EvaluationResult


def calculate_pass_rate(results: List[Union[EvaluationResult, dict]]) -> float:
    """Calculate the pass@1 rate from evaluation results.

    Args:
        results: List of EvaluationResult objects or dictionaries.

    Returns:
        The pass rate as a float between 0 and 1.

    Example:
        >>> results = [
        ...     {"success": True},
        ...     {"success": False},
        ...     {"success": True},
        ... ]
        >>> calculate_pass_rate(results)
        0.6666666666666666
    """
    if not results:
        return 0.0

    successes = sum(
        1
        for r in results
        if (r.success if isinstance(r, EvaluationResult) else r.get("success", False))
    )
    return successes / len(results)


def generate_report(results: List[Union[EvaluationResult, dict]]) -> pd.DataFrame:
    """Generate a summary report from evaluation results.

    Args:
        results: List of EvaluationResult objects or dictionaries.

    Returns:
        A DataFrame with columns: task_id, title, success, and aggregated stats.

    Example:
        >>> results = [...]
        >>> report = generate_report(results)
        >>> print(report)
        >>> print(f"Pass rate: {report['success'].mean():.2%}")
    """
    if not results:
        return pd.DataFrame(columns=["task_id", "title", "success"])

    # Convert EvaluationResults to dicts if needed
    data = [
        r.to_dict() if isinstance(r, EvaluationResult) else r
        for r in results
    ]

    df = pd.DataFrame(data)

    # Ensure required columns exist
    for col in ["task_id", "title", "success"]:
        if col not in df.columns:
            df[col] = None

    return df[["task_id", "title", "success"]]


def print_summary(results: List[Union[EvaluationResult, dict]], llm_name: str = "") -> None:
    """Print a formatted summary of evaluation results.

    Args:
        results: List of evaluation results.
        llm_name: Optional name of the LLM used for display.
    """
    if not results:
        print("No evaluation results to analyze.")
        return

    df = generate_report(results)
    pass_rate = calculate_pass_rate(results)

    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY{f' ({llm_name.upper()})' if llm_name else ''}")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nOverall Pass@1 Rate: {pass_rate:.2%}")
    print("=" * 80)

    # Print failed task details
    failed = [
        r for r in results
        if not (r.success if isinstance(r, EvaluationResult) else r.get("success", False))
    ]

    if failed:
        print("\n--- ANALYSIS OF FAILED TASKS ---\n")
        for r in failed:
            if isinstance(r, EvaluationResult):
                task_id = r.task_id
                log = r.verification_log
            else:
                task_id = r.get("task_id", "unknown")
                log = r.get("verification_log", "No log available")

            print(f"\n----- Details for Failed Task: {task_id} -----")
            print("\n[Verification Log]")
            print(log)
            print("-" * 50)

"""Evaluation pipeline for running multi-task evaluations.

This module provides the main pipeline function for running evaluations
across multiple benchmark tasks.
"""

import time
from pathlib import Path
from typing import Any, List, Optional, Union, TYPE_CHECKING

import pandas as pd

from .task_runner import run_evaluation_on_task, EvaluationResult
from .metrics import calculate_pass_rate, generate_report, print_summary

if TYPE_CHECKING:
    from langchain.agents import AgentExecutor
    from ..benchmarks.base import Benchmark


def run_evaluation_pipeline(
    agent_executor: "AgentExecutor",
    codebase_df: pd.DataFrame,
    task_df: pd.DataFrame,
    project_repo_path: Union[str, Path],
    task_ids: Optional[List[str]] = None,
    start_from_task: Optional[str] = None,
    parser: Optional[Any] = None,
    test_timeout: int = 60,
    delay_between_tasks: float = 2.0,
    print_results: bool = True,
    llm_name: str = "",
) -> List[EvaluationResult]:
    """Run the full evaluation pipeline on multiple tasks.

    This function orchestrates running evaluation on a set of benchmark
    tasks, collecting results, and optionally printing a summary.

    Args:
        agent_executor: The agent executor to use for evaluation.
        codebase_df: DataFrame containing the codebase files.
        task_df: DataFrame containing task definitions.
        project_repo_path: Path to the project repository.
        task_ids: Optional list of specific task IDs to run.
            If None, runs all tasks from task_df.
        start_from_task: Optional task ID to start from (for resuming).
        parser: Optional tree-sitter parser for API guide generation.
        test_timeout: Timeout for running tests in seconds.
        delay_between_tasks: Delay between tasks in seconds (for rate limiting).
        print_results: Whether to print the summary after evaluation.
        llm_name: Name of the LLM for display in summary.

    Returns:
        A list of EvaluationResult objects.

    Example:
        >>> from codeagent.agents import create_agent_executor
        >>> results = run_evaluation_pipeline(
        ...     agent_executor=agent,
        ...     codebase_df=codebase_df,
        ...     task_df=task_df,
        ...     project_repo_path=Path("./mini_transformers_repo"),
        ...     task_ids=["miniformer-01", "miniformer-02"],
        ... )
        >>> print(f"Pass rate: {calculate_pass_rate(results):.2%}")
    """
    project_repo_path = Path(project_repo_path)

    # Determine which tasks to run
    if task_ids is not None:
        tasks_to_run = task_ids
    else:
        tasks_to_run = task_df["task_id"].tolist()

    # Handle start_from_task for resuming
    if start_from_task is not None:
        if start_from_task in tasks_to_run:
            start_idx = tasks_to_run.index(start_from_task)
            tasks_to_run = tasks_to_run[start_idx:]
            print(f"Resuming evaluation from task '{start_from_task}' onwards.")
        else:
            print(f"Warning: Task '{start_from_task}' not found. Running all tasks.")

    print("\n" + "=" * 80)
    print(f"STARTING EVALUATION PIPELINE{f' ({llm_name.upper()})' if llm_name else ''}")
    print(f"Tasks to evaluate: {len(tasks_to_run)}")
    print("=" * 80)

    results: List[EvaluationResult] = []

    for i, task_id in enumerate(tasks_to_run):
        try:
            result = run_evaluation_on_task(
                task_id=task_id,
                agent_executor=agent_executor,
                codebase_df=codebase_df,
                task_df=task_df,
                project_repo_path=project_repo_path,
                parser=parser,
                test_timeout=test_timeout,
            )
            results.append(result)

        except Exception as e:
            print(f"\nError evaluating task {task_id}: {e}")
            results.append(
                EvaluationResult(
                    task_id=task_id,
                    title=f"Task {task_id}",
                    success=False,
                    final_answer="",
                    verification_log=f"FAIL: Pipeline exception: {e}",
                )
            )

        # Rate limiting delay between tasks
        if i < len(tasks_to_run) - 1 and delay_between_tasks > 0:
            print(f"Pausing for {delay_between_tasks}s before next task...")
            time.sleep(delay_between_tasks)

    # Print summary if requested
    if print_results:
        print_summary(results, llm_name)

    return results


def run_evaluation_from_benchmark(
    agent_executor: "AgentExecutor",
    benchmark: "Benchmark",
    project_repo_path: Union[str, Path],
    task_ids: Optional[List[str]] = None,
    parser: Optional[Any] = None,
    test_timeout: int = 60,
    delay_between_tasks: float = 2.0,
    print_results: bool = True,
    llm_name: str = "",
) -> List[EvaluationResult]:
    """Run evaluation using a Benchmark object.

    A convenience wrapper that extracts dataframes from a Benchmark instance.

    Args:
        agent_executor: The agent executor to use.
        benchmark: A Benchmark instance with loaded data.
        project_repo_path: Path to the project repository.
        task_ids: Optional list of specific task IDs to run.
        parser: Optional tree-sitter parser.
        test_timeout: Timeout for running tests in seconds.
        delay_between_tasks: Delay between tasks in seconds.
        print_results: Whether to print the summary.
        llm_name: Name of the LLM for display.

    Returns:
        A list of EvaluationResult objects.

    Example:
        >>> from codeagent.benchmarks import MiniTransformersBench
        >>> benchmark = MiniTransformersBench()
        >>> results = run_evaluation_from_benchmark(
        ...     agent_executor=agent,
        ...     benchmark=benchmark,
        ...     project_repo_path=Path("./repo"),
        ... )
    """
    codebase_df = benchmark.load_codebase()
    task_df = benchmark.load_tasks()

    return run_evaluation_pipeline(
        agent_executor=agent_executor,
        codebase_df=codebase_df,
        task_df=task_df,
        project_repo_path=project_repo_path,
        task_ids=task_ids,
        parser=parser,
        test_timeout=test_timeout,
        delay_between_tasks=delay_between_tasks,
        print_results=print_results,
        llm_name=llm_name,
    )


def compare_results(
    agent_results: List[Union[EvaluationResult, dict]],
    baseline_results: List[dict],
    agent_name: str = "Agent",
    baseline_name: str = "NoAgent",
) -> pd.DataFrame:
    """Compare agent results with baseline results.

    Args:
        agent_results: Results from agent evaluation.
        baseline_results: Results from baseline (no-agent) evaluation.
        agent_name: Name for the agent column.
        baseline_name: Name for the baseline column.

    Returns:
        A DataFrame comparing pass rates and per-task results.
    """
    # Convert to dicts if needed
    agent_dicts = [
        r.to_dict() if isinstance(r, EvaluationResult) else r
        for r in agent_results
    ]

    # Build comparison dataframe
    agent_df = pd.DataFrame(agent_dicts)[["task_id", "title", "success"]]
    agent_df = agent_df.rename(columns={"success": f"{agent_name}_success"})

    baseline_df = pd.DataFrame(baseline_results)[["task_id", "success"]]
    baseline_df = baseline_df.rename(columns={"success": f"{baseline_name}_success"})

    comparison = agent_df.merge(baseline_df, on="task_id", how="outer")

    # Calculate pass rates
    agent_pass_rate = calculate_pass_rate(agent_results)
    baseline_pass_rate = calculate_pass_rate(baseline_results)

    print("\n" + "=" * 80)
    print("COMPARISON: AGENT vs BASELINE")
    print("=" * 80)
    print(comparison.to_string(index=False))
    print(f"\n{agent_name} Pass Rate: {agent_pass_rate:.2%}")
    print(f"{baseline_name} Pass Rate: {baseline_pass_rate:.2%}")
    print(f"Improvement: {(agent_pass_rate - baseline_pass_rate):.2%}")
    print("=" * 80)

    return comparison

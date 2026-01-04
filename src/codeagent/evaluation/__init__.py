"""Evaluation module for CodeAgent.

This module provides evaluation infrastructure:
- Task runner for single task evaluation
- Repository setup for each task
- Metrics and reporting
- Evaluation pipeline for multi-task runs
- HumanEval benchmark integration
- No-agent baseline evaluation
"""

from .task_runner import (
    run_evaluation_on_task,
    EvaluationResult,
    TASK_TO_TEST_MAPPING,
)
from .repository_setup import setup_repository_for_task
from .metrics import (
    calculate_pass_rate,
    generate_report,
    print_summary,
)
from .pipeline import (
    run_evaluation_pipeline,
    run_evaluation_from_benchmark,
    compare_results,
)
from .no_agent_baseline import (
    build_no_agent_prompt,
    run_no_agent_evaluation_on_task,
    run_no_agent_baseline,
)
from .human_eval import (
    clean_humaneval_code,
    run_humaneval_evaluation,
    save_humaneval_samples,
    evaluate_humaneval_results,
    run_full_humaneval_pipeline,
)

__all__ = [
    # Task runner
    "run_evaluation_on_task",
    "EvaluationResult",
    "TASK_TO_TEST_MAPPING",
    # Repository setup
    "setup_repository_for_task",
    # Metrics
    "calculate_pass_rate",
    "generate_report",
    "print_summary",
    # Pipeline
    "run_evaluation_pipeline",
    "run_evaluation_from_benchmark",
    "compare_results",
    # No-agent baseline
    "build_no_agent_prompt",
    "run_no_agent_evaluation_on_task",
    "run_no_agent_baseline",
    # HumanEval
    "clean_humaneval_code",
    "run_humaneval_evaluation",
    "save_humaneval_samples",
    "evaluate_humaneval_results",
    "run_full_humaneval_pipeline",
]

"""Task runner for single task evaluation.

This module provides the core function for running evaluation on
a single benchmark task.
"""

import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING

import pandas as pd

from .repository_setup import setup_repository_for_task

if TYPE_CHECKING:
    from langchain.agents import AgentExecutor


# Task to test function mapping for MiniTransformers benchmark
TASK_TO_TEST_MAPPING: Dict[str, str] = {
    "miniformer-01": "test_config_bias_field",
    "miniformer-02": "test_to_numpy_conversion",
    "miniformer-03": "test_config_to_dict_method",
    "miniformer-04": "test_swish_activation",
    "miniformer-05": "test_positional_embedding_layer",
    # Refactoring tasks: run the whole integration suite
    "miniformer-06": "tests/test_integration.py",
    "miniformer-07": "tests/test_integration.py",
    # Additive/Fix tasks continue
    "miniformer-08": "test_relu_activation",
    "miniformer-09": "test_block_summary_method",
    "miniformer-10": "test_full_model_instantiation",
    "miniformer-11": "test_config_instantiation",
    "miniformer-12": "test_flash_attention_placeholder",
    # Another refactoring task
    "miniformer-13": "tests/test_integration.py",
    # Additive/Fix tasks continue
    "miniformer-14": "test_xavier_initializer",
    "miniformer-15": "test_language_model_head",
}


@dataclass
class EvaluationResult:
    """Result of a single task evaluation.

    Attributes:
        task_id: The task identifier.
        title: The task title.
        success: Whether the task passed verification.
        final_answer: The agent's final answer.
        verification_log: Details of the verification process.
    """

    task_id: str
    title: str
    success: bool
    final_answer: str
    verification_log: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "success": self.success,
            "final_answer": self.final_answer,
            "verification_log": self.verification_log,
        }


def run_evaluation_on_task(
    task_id: str,
    agent_executor: "AgentExecutor",
    codebase_df: pd.DataFrame,
    task_df: pd.DataFrame,
    project_repo_path: Path,
    parser: Any = None,
    test_timeout: int = 60,
) -> EvaluationResult:
    """Run a full evaluation cycle for a single task.

    This function:
    1. Sets up the repository for the task
    2. Runs the agent with the task prompt
    3. Verifies the result using pytest

    Args:
        task_id: The ID of the task to evaluate.
        agent_executor: The agent executor to use.
        codebase_df: DataFrame containing the codebase files.
        task_df: DataFrame containing task definitions.
        project_repo_path: Path to the project repository.
        parser: Optional tree-sitter parser.
        test_timeout: Timeout for running tests in seconds.

    Returns:
        An EvaluationResult with the task outcome.

    Example:
        >>> result = run_evaluation_on_task(
        ...     "miniformer-01",
        ...     agent_executor,
        ...     codebase_df,
        ...     task_df,
        ...     Path("./mini_transformers_repo"),
        ... )
        >>> print(f"Task {result.task_id}: {'PASS' if result.success else 'FAIL'}")
    """
    task_info = task_df[task_df["task_id"] == task_id].iloc[0]

    print("\n" + "=" * 80)
    print(f"STARTING EVALUATION FOR TASK: {task_id} ({task_info['title']})")
    print("=" * 80)

    # Set up the repository
    setup_repository_for_task(
        task_id, codebase_df, task_df, project_repo_path, parser
    )

    # Get the agent prompt
    agent_prompt = task_info["comment"]
    print("\n[Agent's Task/Prompt]:")
    print(textwrap.fill(agent_prompt, width=80))
    print("\n--- AGENT EXECUTION LOG ---")

    # Run the agent
    try:
        result = agent_executor.invoke({"input": agent_prompt})
        final_answer = result.get("output", "Error: No final answer was produced.")
    except Exception as e:
        final_answer = f"CRITICAL AGENT FAILURE: The agent executor raised an exception: {e}"

    print("--- AGENT EXECUTION FINISHED ---\n")

    # Verify the result
    success, verification_log = _verify_task(
        task_id, task_info, project_repo_path, test_timeout
    )

    print(f"[Agent's Final Answer]:\n{textwrap.fill(final_answer, width=80)}\n")
    print(f"[Verification Result]: {verification_log}")
    print(f"--- TASK {task_id} | FINAL OUTCOME: {'PASS' if success else 'FAIL'} ---")
    print("=" * 80)

    return EvaluationResult(
        task_id=task_id,
        title=task_info["title"],
        success=success,
        final_answer=final_answer,
        verification_log=verification_log,
    )


def _verify_task(
    task_id: str,
    task_info: pd.Series,
    project_repo_path: Path,
    test_timeout: int,
) -> tuple[bool, str]:
    """Verify the task result using pytest.

    Args:
        task_id: The task identifier.
        task_info: The task information row.
        project_repo_path: Path to the project repository.
        test_timeout: Timeout for running tests.

    Returns:
        A tuple of (success, verification_log).
    """
    try:
        test_file_to_run = project_repo_path / task_info["test_file_path"]
        test_specifier = TASK_TO_TEST_MAPPING.get(task_id)

        if not test_specifier:
            return False, f"No test mapping for task_id '{task_id}'."

        # Determine the pytest command
        if test_specifier.endswith(".py"):
            # Run the entire test file (for refactoring tasks)
            pytest_command = [
                "python", "-m", "pytest",
                str(project_repo_path / test_specifier),
            ]
            verification_target = f"Full suite '{test_specifier}'"
        else:
            # Run a specific test function
            pytest_command = [
                "python", "-m", "pytest",
                str(test_file_to_run),
                "-k", test_specifier,
            ]
            verification_target = f"Isolated test '{test_specifier}'"

        # Set up the environment
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{project_repo_path}:{env.get('PYTHONPATH', '')}"

        # Run the tests
        process = subprocess.run(
            pytest_command,
            cwd=str(project_repo_path),
            capture_output=True,
            text=True,
            timeout=test_timeout,
            env=env,
        )

        # Check for success
        if (
            process.returncode == 0
            and "failed" not in process.stdout.lower()
            and "error" not in process.stdout.lower()
        ):
            return True, f"PASS: {verification_target} passed."
        else:
            return False, (
                f"FAIL: {verification_target} failed.\n"
                f"--- STDOUT ---\n{process.stdout}\n"
                f"--- STDERR ---\n{process.stderr}"
            )

    except subprocess.TimeoutExpired:
        return False, f"FAIL: Test execution timed out after {test_timeout} seconds."
    except Exception as e:
        return False, f"FAIL: An exception occurred during verification: {e}"

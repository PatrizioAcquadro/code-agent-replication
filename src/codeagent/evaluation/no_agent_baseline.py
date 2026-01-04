"""No-agent baseline evaluation.

This module provides functions for running baseline evaluations
without the agent loop, using only the base LLM.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .repository_setup import setup_repository_for_task
from .task_runner import TASK_TO_TEST_MAPPING


def build_no_agent_prompt(
    task_info: pd.Series,
    codebase_df: pd.DataFrame,
) -> str:
    """Create a prompt for baseline LLM evaluation without tools.

    This creates a single, large prompt by providing all necessary context,
    including the code of the file to be modified.

    Args:
        task_info: A pandas Series containing task information.
        codebase_df: DataFrame containing the codebase files.

    Returns:
        A formatted prompt string for the LLM.

    Example:
        >>> task_info = task_df[task_df['task_id'] == 'miniformer-01'].iloc[0]
        >>> prompt = build_no_agent_prompt(task_info, codebase_df)
        >>> print(prompt[:100])
    """
    target_file_path = task_info["class_link"]
    existing_code = "This is a new file that you must create."

    file_row = codebase_df[codebase_df["path"] == target_file_path]
    if not file_row.empty:
        content = file_row.iloc[0]["content"]
        if isinstance(content, list):
            existing_code = "".join(content)
        else:
            existing_code = content

    return f"""
You are an expert Python programmer. Your task is to generate the complete, final source code for a single Python file based on the user's request.
Your output MUST be ONLY the raw Python code, without any surrounding text, conversation, or markdown fences like ```python.

### USER REQUEST:
{task_info['comment']}

### CONTEXT: EXISTING CODE IN `{target_file_path}`
(If the request is to create a new file, this section will state that.)
```python
{existing_code}
```

Based on the request and the existing code, provide the complete and final version of the code for the file `{target_file_path}`.
"""


def run_no_agent_evaluation_on_task(
    task_id: str,
    llm_instance: Any,
    codebase_df: pd.DataFrame,
    task_df: pd.DataFrame,
    project_repo_path: Union[str, Path],
    parser: Optional[Any] = None,
    test_timeout: int = 60,
) -> Dict[str, Any]:
    """Run evaluation for a single task using the base LLM without tools.

    This function runs a full evaluation cycle without the agent loop,
    providing a baseline for comparison.

    Args:
        task_id: The ID of the task to evaluate.
        llm_instance: The LLM instance to use for generation.
        codebase_df: DataFrame containing the codebase files.
        task_df: DataFrame containing task definitions.
        project_repo_path: Path to the project repository.
        parser: Optional tree-sitter parser.
        test_timeout: Timeout for running tests in seconds.

    Returns:
        A dictionary with task_id, title, success, and verification_log.

    Example:
        >>> result = run_no_agent_evaluation_on_task(
        ...     "miniformer-01",
        ...     llm,
        ...     codebase_df,
        ...     task_df,
        ...     Path("./mini_transformers_repo"),
        ... )
        >>> print(f"Task {result['task_id']}: {'PASS' if result['success'] else 'FAIL'}")
    """
    from ..utils.file_ops import write_file_content
    from ..utils.code_cleaning import clean_final_code

    project_repo_path = Path(project_repo_path)
    task_info = task_df[task_df["task_id"] == task_id].iloc[0]
    target_file_path = task_info["class_link"]

    print("\n" + "=" * 80)
    print(f"STARTING 'NOAGENT' EVALUATION FOR TASK: {task_id}")
    print("=" * 80)

    # Set up the repository
    setup_repository_for_task(task_id, codebase_df, task_df, project_repo_path, parser)

    # Build the prompt
    prompt = build_no_agent_prompt(task_info, codebase_df)

    # Invoke the LLM
    print("--- Invoking Base LLM... ---")
    try:
        response = llm_instance.invoke(prompt)
        raw_llm_output = (
            response.content if hasattr(response, "content") else str(response)
        )
    except Exception as e:
        raw_llm_output = f"CRITICAL LLM FAILURE: {e}"
    print("--- Base LLM invocation complete. ---\n")

    print(f"--- Base LLM Raw Output for Task {task_id} ---")
    print(raw_llm_output[:500] + "..." if len(raw_llm_output) > 500 else raw_llm_output)
    print("------------------------------------------")

    # Clean the code and check if the LLM actually produced code
    final_code = clean_final_code(raw_llm_output)

    # Only write the file if the LLM produced valid code
    if final_code and "CRITICAL" not in raw_llm_output:
        write_file_content(target_file_path, final_code, project_repo_path)
    else:
        print(
            f"LLM did not produce valid code for {target_file_path}. "
            "The file will not be written."
        )
        return {
            "task_id": task_id,
            "title": task_info["title"],
            "success": False,
            "verification_log": (
                "FAIL: LLM did not generate any valid code to write to the file."
            ),
        }

    # Verify the result
    success, verification_log = _verify_no_agent_task(
        task_id, task_info, project_repo_path, test_timeout
    )

    print(f"\n[Verification Result]: {verification_log}")
    print(f"--- TASK {task_id} | FINAL OUTCOME: {'PASS' if success else 'FAIL'} ---")

    return {
        "task_id": task_id,
        "title": task_info["title"],
        "success": success,
        "verification_log": verification_log,
    }


def _verify_no_agent_task(
    task_id: str,
    task_info: pd.Series,
    project_repo_path: Path,
    test_timeout: int,
) -> tuple[bool, str]:
    """Verify the no-agent task result using pytest.

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
            pytest_command = [
                "python", "-m", "pytest",
                str(project_repo_path / test_specifier),
            ]
        else:
            pytest_command = [
                "python", "-m", "pytest",
                str(test_file_to_run),
                "-k", test_specifier,
            ]

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
            return True, f"PASS: Test '{test_specifier}' passed."
        else:
            return False, (
                f"FAIL: Test '{test_specifier}' failed.\n"
                f"--- STDOUT ---\n{process.stdout}\n"
                f"--- STDERR ---\n{process.stderr}"
            )

    except subprocess.TimeoutExpired:
        return False, f"FAIL: Test execution timed out after {test_timeout} seconds."
    except Exception as e:
        return False, f"FAIL: An exception occurred during verification: {e}"


def run_no_agent_baseline(
    llm_instance: Any,
    codebase_df: pd.DataFrame,
    task_df: pd.DataFrame,
    project_repo_path: Union[str, Path],
    task_ids: Optional[List[str]] = None,
    parser: Optional[Any] = None,
    delay_between_tasks: float = 2.0,
) -> List[Dict[str, Any]]:
    """Run no-agent baseline evaluation on multiple tasks.

    Args:
        llm_instance: The LLM instance to use.
        codebase_df: DataFrame containing the codebase files.
        task_df: DataFrame containing task definitions.
        project_repo_path: Path to the project repository.
        task_ids: Optional list of task IDs to run. If None, runs all tasks.
        parser: Optional tree-sitter parser.
        delay_between_tasks: Delay in seconds between tasks (for rate limiting).

    Returns:
        A list of result dictionaries.
    """
    import time

    if task_ids is None:
        task_ids = task_df["task_id"].tolist()

    results = []
    for i, task_id in enumerate(task_ids):
        result = run_no_agent_evaluation_on_task(
            task_id=task_id,
            llm_instance=llm_instance,
            codebase_df=codebase_df,
            task_df=task_df,
            project_repo_path=project_repo_path,
            parser=parser,
        )
        results.append(result)

        # Rate limiting delay
        if i < len(task_ids) - 1 and delay_between_tasks > 0:
            time.sleep(delay_between_tasks)

    return results

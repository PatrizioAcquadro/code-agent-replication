"""HumanEval benchmark integration.

This module provides functions for running the HumanEval benchmark
for function-level code generation evaluation.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.agents import AgentExecutor


def clean_humaneval_code(code_string: str) -> str:
    """Extract raw Python code from an LLM's response for HumanEval.

    This function handles the specific format requirements for HumanEval
    where we need just the function body, not the signature.

    Args:
        code_string: The raw LLM output.

    Returns:
        Cleaned Python code suitable for HumanEval evaluation.
    """
    if not code_string:
        return ""

    # Strip the 'Final Answer:' prefix first
    cleaned_string = code_string.replace("Final Answer:", "").strip()

    # Extract code from markdown fences
    match = re.search(r"```(?:python\n)?(.*)```", cleaned_string, re.DOTALL)
    if match:
        code_block = match.group(1).strip()
        # If the agent includes the 'def' line, try to strip it
        if code_block.lstrip().startswith("def "):
            try:
                return code_block.split(":", 1)[1].strip()
            except IndexError:
                return code_block
        return code_block

    # Fallback for responses without markdown
    if cleaned_string.lstrip().startswith("def "):
        try:
            return cleaned_string.split(":", 1)[1].strip()
        except IndexError:
            return cleaned_string

    return cleaned_string


def run_humaneval_evaluation(
    agent_executor: "AgentExecutor",
    problems_dict: Dict[str, Dict[str, Any]],
    num_samples: Optional[int] = None,
    delay_between_tasks: float = 5.0,
) -> List[Dict[str, str]]:
    """Run the agent on HumanEval problems and collect completions.

    Args:
        agent_executor: The agent executor to use.
        problems_dict: Dictionary of HumanEval problems from read_problems().
        num_samples: Number of problems to run. If None, runs all.
        delay_between_tasks: Delay in seconds between tasks (for rate limiting).

    Returns:
        A list of dictionaries with task_id and completion fields.

    Example:
        >>> from human_eval.data import read_problems
        >>> problems = read_problems()
        >>> completions = run_humaneval_evaluation(agent, problems, num_samples=15)
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x  # Fallback if tqdm not available

    if not agent_executor or not problems_dict:
        print("Agent or problems not available. Skipping evaluation.")
        return []

    task_ids = list(problems_dict.keys())
    if num_samples is not None:
        task_ids = task_ids[:num_samples]

    completions = []

    for i, task_id in enumerate(tqdm(task_ids, desc="Solving HumanEval Problems")):
        problem = problems_dict[task_id]

        # Construct the prompt
        agent_prompt = (
            "You are an expert Python programmer. Your task is to complete "
            "the function body for the following problem.\n"
            "**IMPORTANT**: The function signature and docstring are already "
            "provided. You must ONLY provide the implementation code (the "
            "indented body).\n"
            "Do not repeat the `def` line or the docstring in your thoughts, "
            "actions, or final answer.\n\n"
            f"--- FUNCTION TO COMPLETE ---\n{problem['prompt']}"
            "--- END FUNCTION ---"
        )

        try:
            result = agent_executor.invoke({"input": agent_prompt})
            generated_body = clean_humaneval_code(
                result.get("output", "# Agent failed to produce code.")
            )

            # Construct the full completion by combining prompt and generated body
            indented_body = "\n".join(
                ["    " + line for line in generated_body.splitlines()]
            )
            full_completion = problem["prompt"] + "\n" + indented_body

            completions.append(dict(task_id=task_id, completion=full_completion))

        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            completions.append(
                dict(
                    task_id=task_id,
                    completion=problem["prompt"]
                    + f"\n    # Agent raised an exception: {e}",
                )
            )

        # Rate limiting
        if i < len(task_ids) - 1 and delay_between_tasks > 0:
            print(f"Pausing for {delay_between_tasks}s to respect API rate limits...")
            time.sleep(delay_between_tasks)

    return completions


def save_humaneval_samples(
    completions: List[Dict[str, str]],
    output_path: Path,
) -> Path:
    """Save HumanEval completions to a JSONL file.

    Args:
        completions: List of completion dictionaries.
        output_path: Path to save the JSONL file.

    Returns:
        The path to the saved file.
    """
    try:
        from human_eval.data import write_jsonl
    except ImportError:
        raise ImportError(
            "human_eval package required. Install with: pip install human_eval"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(str(output_path), completions)
    print(f"Saved {len(completions)} completions to {output_path}")

    return output_path


def evaluate_humaneval_results(
    samples_file: Path,
    k: List[int] = [1],
) -> Dict[str, float]:
    """Evaluate HumanEval completions using the official evaluation.

    Args:
        samples_file: Path to the JSONL file with completions.
        k: List of k values for pass@k metric.

    Returns:
        Dictionary with pass@k results.

    Example:
        >>> results = evaluate_humaneval_results(Path("./samples.jsonl"), k=[1, 10])
        >>> print(f"Pass@1: {results['pass@1']:.2%}")
    """
    try:
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        raise ImportError(
            "human_eval package required. Install with: pip install human_eval"
        )

    results = evaluate_functional_correctness(str(samples_file), k=k)
    return results


def run_full_humaneval_pipeline(
    agent_executor: "AgentExecutor",
    output_dir: Path,
    num_samples: Optional[int] = None,
    llm_name: str = "agent",
    delay_between_tasks: float = 5.0,
) -> Dict[str, Any]:
    """Run the full HumanEval evaluation pipeline.

    This function:
    1. Loads HumanEval problems
    2. Runs the agent on problems
    3. Saves completions
    4. Evaluates and returns results

    Args:
        agent_executor: The agent executor to use.
        output_dir: Directory to save results.
        num_samples: Number of problems to run. If None, runs all.
        llm_name: Name for labeling output files.
        delay_between_tasks: Delay between tasks in seconds.

    Returns:
        Dictionary with completions, results, and file paths.
    """
    try:
        from human_eval.data import read_problems
    except ImportError:
        raise ImportError(
            "human_eval package required. Install with: pip install human_eval"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load problems
    print("Loading HumanEval problems...")
    problems = read_problems()
    print(f"Loaded {len(problems)} problems.")

    # Run evaluation
    print(f"\nRunning evaluation with {llm_name}...")
    completions = run_humaneval_evaluation(
        agent_executor=agent_executor,
        problems_dict=problems,
        num_samples=num_samples,
        delay_between_tasks=delay_between_tasks,
    )

    # Save completions
    samples_file = output_dir / f"humaneval_samples_{llm_name}.jsonl"
    save_humaneval_samples(completions, samples_file)

    # Evaluate
    print("\nEvaluating completions...")
    results = evaluate_humaneval_results(samples_file)

    # Print summary
    print("\n" + "=" * 80)
    print(f"HUMANEVAL RESULTS ({llm_name.upper()})")
    print("=" * 80)
    for metric, value in results.items():
        print(f"{metric}: {value:.2%}")
    print("=" * 80)

    return {
        "completions": completions,
        "results": results,
        "samples_file": samples_file,
    }

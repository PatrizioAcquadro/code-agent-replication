"""Code analysis utilities for benchmarks.

This module provides functions to analyze Python code for metrics
like import counts, symbol definitions, and task classification.
"""

import ast
import re
from typing import Dict, List, Optional, Any

import pandas as pd


def force_string(content: Any) -> str:
    """Ensure content is a single string.

    Handles the case where content might be a list of strings.

    Args:
        content: The content to normalize.

    Returns:
        A single string representation.

    Example:
        >>> force_string(["line1", "line2"])
        'line1\\nline2'
        >>> force_string("single string")
        'single string'
    """
    if isinstance(content, list):
        return "\n".join(map(str, content))
    return str(content) if content is not None else ""


def extract_imports(code: str, internal_prefix: str = "numpy_ml") -> Dict[str, List[str]]:
    """Extract import statements from Python code.

    Categorizes imports as internal (within the project) or external.

    Args:
        code: The Python source code to analyze.
        internal_prefix: The prefix for internal module imports.

    Returns:
        A dictionary with 'internal' and 'external' keys,
        each containing a list of module names.

    Example:
        >>> code = '''
        ... import numpy
        ... from numpy_ml.layers import Layer
        ... '''
        >>> imports = extract_imports(code)
        >>> imports['external']
        ['numpy']
        >>> imports['internal']
        ['numpy_ml.layers']
    """
    imports: Dict[str, List[str]] = {"internal": [], "external": []}

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(internal_prefix):
                        imports["internal"].append(alias.name)
                    else:
                        imports["external"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.module.startswith(internal_prefix):
                        imports["internal"].append(node.module)
                    else:
                        imports["external"].append(node.module)
    except SyntaxError:
        pass

    return imports


def analyze_code_symbols(code: str) -> pd.Series:
    """Analyze Python code to count classes and functions.

    Args:
        code: The Python source code to analyze.

    Returns:
        A pandas Series with 'class_count' and 'function_count'.

    Example:
        >>> code = '''
        ... class Foo:
        ...     def bar(self):
        ...         pass
        ... def baz():
        ...     pass
        ... '''
        >>> counts = analyze_code_symbols(code)
        >>> counts['class_count']
        1
        >>> counts['function_count']
        2
    """
    functions = 0
    classes = 0

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions += 1
            elif isinstance(node, ast.ClassDef):
                classes += 1
    except SyntaxError:
        pass

    return pd.Series(
        [classes, functions],
        index=["class_count", "function_count"],
    )


def get_task_type(title: str) -> str:
    """Determine the task type based on its title.

    Uses a heuristic: PascalCase titles are classes, others are functions.

    Args:
        title: The task title.

    Returns:
        Either 'Class' or 'Function'.

    Example:
        >>> get_task_type("TransformerConfig")
        'Class'
        >>> get_task_type("calculate_loss")
        'Function'
    """
    if isinstance(title, str) and re.match(r"^[A-Z][a-zA-Z0-9]+$", title):
        return "Class"
    return "Function"


def get_target_path_from_annotation(annotation: str) -> Optional[str]:
    """Extract the file path from a class annotation.

    Converts a dotted annotation path to a file path.

    Args:
        annotation: The class annotation (e.g., 'numpy_ml.ngram.AdditiveNGram').

    Returns:
        The file path (e.g., 'numpy_ml/ngram.py'), or None if invalid.

    Example:
        >>> get_target_path_from_annotation("numpy_ml.ngram.AdditiveNGram")
        'numpy_ml/ngram.py'
    """
    if not isinstance(annotation, str):
        return None
    parts = annotation.split(".")
    if len(parts) < 2:
        return None
    # Remove the class name (last part) and join with /
    return "/".join(parts[:-1]) + ".py"

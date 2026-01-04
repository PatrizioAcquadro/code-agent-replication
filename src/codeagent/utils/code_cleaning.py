"""Code extraction and cleaning utilities.

This module provides functions to extract and clean Python code from
LLM outputs, handling various formats like markdown code blocks.
"""

import re
from typing import Optional


def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from markdown code blocks.

    Handles both fenced code blocks (```python ... ```) and plain
    fenced blocks (``` ... ```).

    Args:
        text: Text potentially containing markdown code blocks.

    Returns:
        The extracted code, or the original text if no blocks found.

    Example:
        >>> text = '''Here is the code:
        ... ```python
        ... def hello():
        ...     print("Hello")
        ... ```
        ... '''
        >>> extract_code_from_markdown(text)
        'def hello():\\n    print("Hello")'
    """
    if not text:
        return ""

    # Try to find ```python ... ``` blocks first
    match = re.search(r"```python\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try plain ``` ... ``` blocks
    match = re.search(r"```\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


def clean_final_code(code_string: Optional[str]) -> str:
    """Clean the agent's final answer to extract only raw Python code.

    This function removes:
    - Markdown fences (```python...```)
    - "Final Answer:" prefixes
    - Surrounding text and conversation

    Args:
        code_string: The raw output from an LLM or agent.

    Returns:
        Clean Python code without markdown or prefixes.

    Example:
        >>> raw = '''Final Answer:
        ... ```python
        ... def add(a, b):
        ...     return a + b
        ... ```
        ... '''
        >>> clean_final_code(raw)
        'def add(a, b):\\n    return a + b'
    """
    if code_string is None:
        return ""

    # Strip the 'Final Answer:' prefix first
    cleaned = code_string.replace("Final Answer:", "").strip()

    # Extract code from markdown fences
    match = re.search(r"```(?:python\n)?(.*)```", cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return as-is if no markdown found
    return cleaned.strip()


def strip_function_signature(code: str) -> str:
    """Strip the function signature from code, keeping only the body.

    Useful for HumanEval-style tasks where only the function body is needed.

    Args:
        code: Python code potentially starting with a function definition.

    Returns:
        Just the function body if a def was found, otherwise the original code.

    Example:
        >>> code = '''def add(a, b):
        ...     return a + b'''
        >>> strip_function_signature(code)
        'return a + b'
    """
    if not code:
        return ""

    code = code.strip()

    # If code starts with 'def', try to extract just the body
    if code.lstrip().startswith("def "):
        try:
            # Find the first colon and return everything after it
            colon_idx = code.index(":")
            body = code[colon_idx + 1:].strip()
            return body
        except (ValueError, IndexError):
            pass

    return code


def ensure_proper_indentation(code: str, indent: str = "    ") -> str:
    """Ensure each line of code has proper indentation.

    Args:
        code: The code to indent.
        indent: The indentation string to use.

    Returns:
        Code with each line properly indented.
    """
    if not code:
        return ""

    lines = code.splitlines()
    indented_lines = [indent + line if line.strip() else line for line in lines]
    return "\n".join(indented_lines)

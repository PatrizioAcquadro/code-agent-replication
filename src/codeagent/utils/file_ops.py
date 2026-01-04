"""Safe file operations for CodeAgent.

This module provides secure file I/O functions that ensure all operations
stay within a defined project boundary, preventing path traversal attacks.
"""

import os
from pathlib import Path
from typing import Union


def safe_join_path(base_path: Union[str, Path], relative_path: str) -> str:
    """Safely join a base path with a relative path.

    This function prevents directory traversal attacks by ensuring the
    resulting path is within the base directory.

    Args:
        base_path: The base directory path.
        relative_path: The relative path to join.

    Returns:
        The safely joined absolute path as a string.

    Raises:
        ValueError: If the resulting path would be outside the base directory.

    Example:
        >>> safe_join_path("/project", "src/main.py")
        '/project/src/main.py'
        >>> safe_join_path("/project", "../etc/passwd")  # Raises ValueError
    """
    base_path = str(base_path)
    normalized_relative = os.path.normpath(
        os.path.join("/", relative_path)
    ).lstrip(os.sep)
    full_path = os.path.join(base_path, normalized_relative)

    if not os.path.realpath(full_path).startswith(os.path.realpath(base_path)):
        raise ValueError(
            f"Security Error: Path traversal attempt detected for '{relative_path}'."
        )
    return full_path


def read_file_content(
    relative_file_path: str,
    base_path: Union[str, Path, None] = None,
) -> str:
    """Read file content from within the project repository.

    Args:
        relative_file_path: Path relative to the base directory.
        base_path: Base directory path. If None, uses current working directory.

    Returns:
        The file contents as a string, or an error message starting with "Error:".

    Example:
        >>> content = read_file_content("src/main.py", "/project")
        >>> if not content.startswith("Error:"):
        ...     print(content)
    """
    if base_path is None:
        base_path = os.getcwd()

    try:
        full_path = safe_join_path(base_path, relative_file_path)
        if not os.path.isfile(full_path):
            return f"Error: File '{relative_file_path}' not found in repository."
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error reading file '{relative_file_path}': {str(e)}"


def write_file_content(
    relative_file_path: str,
    content: str,
    base_path: Union[str, Path, None] = None,
) -> str:
    """Write content to a file within the project repository.

    Creates parent directories if they don't exist.

    Args:
        relative_file_path: Path relative to the base directory.
        content: The content to write.
        base_path: Base directory path. If None, uses current working directory.

    Returns:
        A success message or an error message starting with "Error:".

    Example:
        >>> result = write_file_content("output/result.txt", "Hello", "/project")
        >>> print(result)
        Successfully wrote to 'output/result.txt'.
    """
    if base_path is None:
        base_path = os.getcwd()

    try:
        full_path = safe_join_path(base_path, relative_file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to '{relative_file_path}'."
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error writing to file '{relative_file_path}': {str(e)}"


def list_files_in_repo(
    relative_directory_path: str = ".",
    base_path: Union[str, Path, None] = None,
) -> str:
    """List files in a directory within the project repository.

    Args:
        relative_directory_path: Path relative to the base directory.
        base_path: Base directory path. If None, uses current working directory.

    Returns:
        A formatted string listing directory contents, or an error message.

    Example:
        >>> print(list_files_in_repo("src", "/project"))
        Contents of repository path 'src':
        - main.py (FILE)
        - utils (DIR)
    """
    if base_path is None:
        base_path = os.getcwd()

    try:
        scan_path = safe_join_path(base_path, relative_directory_path)
        if not os.path.isdir(scan_path):
            return f"Error: Directory '{relative_directory_path}' not found in repository."

        items = os.listdir(scan_path)
        if not items:
            return f"Directory '{relative_directory_path}' is empty."

        result_str = f"Contents of repository path '{relative_directory_path}':\n"
        for item in sorted(items):
            item_path = os.path.join(scan_path, item)
            item_type = "DIR" if os.path.isdir(item_path) else "FILE"
            result_str += f"- {item} ({item_type})\n"
        return result_str.strip()
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error listing files in '{relative_directory_path}': {str(e)}"

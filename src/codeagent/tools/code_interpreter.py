"""CodeInterpreterTool: Execute Python code in isolated environment.

This tool executes Python scripts in the project's root directory,
allowing the agent to test code and perform file operations.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Type, Union, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class CodeInterpreterInput(BaseModel):
    """Input schema for the CodeInterpreterTool."""

    code_to_execute: str = Field(
        ...,
        description="The string of Python code to execute in the repository context.",
    )


class CodeInterpreterTool(BaseTool):
    """A tool to execute Python scripts in the project context.

    This tool creates a temporary file with the provided code, executes it
    as a subprocess, and returns the output. It's used for:
    - Creating new files
    - Modifying existing files
    - Running tests
    - Verifying code behavior

    Attributes:
        name: Tool name for agent reference.
        description: Detailed description for the agent.
        args_schema: Pydantic input validation schema.
        project_path: Base path for the project repository.
        timeout: Maximum execution time in seconds.
    """

    name: str = "CodeInterpreter"
    description: str = (
        "Executes a Python script in the project's root directory. "
        "This is your primary tool for all file system modifications and for testing your work.\n"
        "You MUST use this tool to perform any of the following actions:\n"
        "1. **CREATE a new file:** Provide a script that opens a new file path in write mode ('w'). "
        "Example: `with open('path/to/new_file.py', 'w') as f: f.write('# New code')`\n"
        "2. **MODIFY an existing file:** Provide a script that reads the file, modifies the content, "
        "and writes it back.\n"
        "3. **TEST code:** Provide a script that runs `pytest` or other checks to verify your changes. "
        "Example: `import pytest; pytest.main(['tests/test_file.py'])`\n"
        "The tool returns the script's stdout and stderr, which you should use to confirm "
        "that your action was successful."
    )
    args_schema: Type[BaseModel] = CodeInterpreterInput
    project_path: Path = Path(".")
    timeout: int = 20

    def __init__(
        self,
        project_path: Union[str, Path] = ".",
        timeout: int = 20,
        **kwargs: Any,
    ) -> None:
        """Initialize the tool with a project path.

        Args:
            project_path: The base path for the project repository.
            timeout: Maximum execution time in seconds.
            **kwargs: Additional arguments passed to BaseTool.
        """
        super().__init__(**kwargs)
        self.project_path = Path(project_path)
        self.timeout = timeout

    def _run(self, code_to_execute: str) -> str:
        """Execute Python code in the project context.

        Args:
            code_to_execute: The Python code to execute.

        Returns:
            Combined stdout and stderr from the execution, or an error message.
        """
        if not code_to_execute.strip():
            return "Error: No code provided to execute."

        temp_file_path = None
        try:
            # Ensure project path exists
            project_path_str = str(self.project_path)
            os.makedirs(project_path_str, exist_ok=True)

            # Create a temporary file to hold the code
            with tempfile.NamedTemporaryFile(
                mode="w+",
                delete=False,
                suffix=".py",
                dir=project_path_str,
            ) as temp_file:
                temp_file.write(code_to_execute)
                temp_file_path = temp_file.name

            # Execute the file as a subprocess
            process = subprocess.run(
                ["python", temp_file_path],
                cwd=project_path_str,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            # Combine stdout and stderr
            output = ""
            if process.stdout:
                output += f"--- STDOUT ---\n{process.stdout}\n"
            if process.stderr:
                output += f"--- STDERR ---\n{process.stderr}\n"

            return output if output else "Script executed with no output."

        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out after {self.timeout} seconds."

        except Exception as e:
            return f"An unexpected error occurred in the CodeInterpreterTool: {e}"

        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

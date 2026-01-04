"""FormatCheckTool: Python code formatting using black.

This tool uses the black formatter to ensure code style consistency
and validate Python syntax.
"""

from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import black


class FormatCheckInput(BaseModel):
    """Input schema for the FormatCheckTool."""

    code_to_format: str = Field(
        description="The Python code string that needs to be formatted."
    )


class FormatCheckTool(BaseTool):
    """A tool to format Python code using the black formatter.

    This tool validates Python syntax and applies consistent formatting.
    It should be used as a final step before writing code to ensure
    code quality and style consistency.

    Attributes:
        name: Tool name for agent reference.
        description: Detailed description for the agent.
        args_schema: Pydantic input validation schema.
    """

    name: str = "FormatCheck"
    description: str = (
        "A utility to automatically format Python code using the 'black' standard. "
        "**This is a useful final step to ensure code quality and style consistency "
        "before writing the code to a file.** "
        "It takes the raw Python code as a string and returns the formatted code "
        "as a string. This tool will fail if the input code has a syntax error."
    )
    args_schema: Type[BaseModel] = FormatCheckInput

    def _run(self, code_to_format: str) -> str:
        """Format Python code using black.

        Args:
            code_to_format: The Python code to format.

        Returns:
            The formatted code, or an error message if formatting fails.
        """
        if not code_to_format or not code_to_format.strip():
            return "Error: No code provided to format."

        try:
            # First validate syntax
            compile(code_to_format, "<string>", "exec")

            # Then format with black
            formatted_code = black.format_str(
                code_to_format,
                mode=black.Mode(line_length=88),
            )
            return formatted_code

        except black.NothingChanged:
            # Code was already properly formatted
            return code_to_format

        except SyntaxError as e:
            return f"Error during formatting: invalid syntax - {e}"

        except Exception as e:
            return f"Error during formatting: {str(e)}"

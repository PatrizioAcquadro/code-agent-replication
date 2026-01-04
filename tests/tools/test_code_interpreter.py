"""Tests for the CodeInterpreterTool."""

import pytest

from codeagent.tools import CodeInterpreterTool, CodeInterpreterInput


class TestCodeInterpreterTool:
    """Tests for the CodeInterpreterTool."""

    @pytest.fixture
    def tool(self, temp_project_dir):
        """Create a CodeInterpreterTool instance."""
        return CodeInterpreterTool(project_path=temp_project_dir)

    def test_tool_has_correct_name(self, tool):
        """Test that the tool has the expected name."""
        assert tool.name == "CodeInterpreter"

    def test_tool_has_description(self, tool):
        """Test that the tool has a description."""
        assert len(tool.description) > 0
        assert "execute" in tool.description.lower() or "run" in tool.description.lower()

    def test_execute_simple_code(self, tool):
        """Test executing simple Python code."""
        code = "print('Hello, World!')"
        result = tool._run(code)

        assert "Hello, World!" in result

    def test_execute_arithmetic(self, tool):
        """Test executing arithmetic operations."""
        code = "print(2 + 3 * 4)"
        result = tool._run(code)

        assert "14" in result

    def test_execute_with_error(self, tool):
        """Test handling code that raises an error."""
        code = "raise ValueError('Test error')"
        result = tool._run(code)

        # Should contain error information
        assert "error" in result.lower() or "ValueError" in result

    def test_execute_syntax_error(self, tool):
        """Test handling code with syntax errors."""
        code = "def broken( syntax"
        result = tool._run(code)

        assert "error" in result.lower() or "syntax" in result.lower()

    def test_execute_multiline_code(self, tool):
        """Test executing multiline code."""
        code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
'''
        result = tool._run(code)
        assert "120" in result

    def test_execute_with_imports(self, tool):
        """Test executing code with standard library imports."""
        code = '''
import math
print(math.sqrt(16))
'''
        result = tool._run(code)
        assert "4" in result

    def test_execute_empty_code(self, tool):
        """Test handling empty code."""
        result = tool._run("")
        # Should handle gracefully
        assert result is not None

    def test_input_schema(self):
        """Test that the input schema is properly defined."""
        schema = CodeInterpreterInput
        assert hasattr(schema, 'model_fields')
        assert 'code' in schema.model_fields

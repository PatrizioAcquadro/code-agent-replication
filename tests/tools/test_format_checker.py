"""Tests for the FormatCheckTool."""

import pytest

from codeagent.tools import FormatCheckTool, FormatCheckInput


class TestFormatCheckTool:
    """Tests for the FormatCheckTool."""

    @pytest.fixture
    def tool(self):
        """Create a FormatCheckTool instance."""
        return FormatCheckTool()

    def test_tool_has_correct_name(self, tool):
        """Test that the tool has the expected name."""
        assert tool.name == "FormatCheck"

    def test_tool_has_description(self, tool):
        """Test that the tool has a description."""
        assert len(tool.description) > 0
        assert "format" in tool.description.lower() or "black" in tool.description.lower()

    def test_format_valid_code(self, tool):
        """Test formatting valid Python code."""
        code = "def foo():return 42"
        result = tool._run(code)

        assert "def foo():" in result
        assert "return 42" in result

    def test_format_already_formatted(self, tool):
        """Test code that is already properly formatted."""
        code = '''def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        result = tool._run(code)
        # Should return the same or similar code
        assert "def hello" in result
        assert "return" in result

    def test_format_invalid_syntax(self, tool):
        """Test handling of code with invalid syntax."""
        code = "def foo( invalid syntax here"
        result = tool._run(code)

        # Should return an error message
        assert "error" in result.lower() or "cannot" in result.lower()

    def test_format_empty_code(self, tool):
        """Test handling of empty code."""
        result = tool._run("")
        # Should handle gracefully
        assert result is not None

    def test_format_multiline_code(self, tool):
        """Test formatting multiline code."""
        code = '''def calculate(a,b,c):
    x=a+b
    y=b+c
    return x*y'''
        result = tool._run(code)

        assert "def calculate" in result
        assert "return" in result

    def test_input_schema(self):
        """Test that the input schema is properly defined."""
        schema = FormatCheckInput
        assert hasattr(schema, 'model_fields')
        assert 'code' in schema.model_fields

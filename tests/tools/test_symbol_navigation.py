"""Tests for the CodeSymbolNavigationTool."""

import pytest

from codeagent.tools import CodeSymbolNavigationTool, SymbolNavigationInput


class TestCodeSymbolNavigationTool:
    """Tests for the CodeSymbolNavigationTool."""

    @pytest.fixture
    def tool(self, sample_codebase):
        """Create a CodeSymbolNavigationTool instance."""
        return CodeSymbolNavigationTool(project_path=sample_codebase)

    def test_tool_has_correct_name(self, tool):
        """Test that the tool has the expected name."""
        assert tool.name == "CodeSymbolNavigation"

    def test_tool_has_description(self, tool):
        """Test that the tool has a description."""
        assert len(tool.description) > 0
        assert "symbol" in tool.description.lower() or "code" in tool.description.lower()

    def test_find_class_symbol(self, tool):
        """Test finding a class definition."""
        result = tool._run("BaseModel")

        # Should find the class
        assert "BaseModel" in result or "class" in result.lower()

    def test_find_function_symbol(self, tool):
        """Test finding a function definition."""
        result = tool._run("validate_email")

        # Should find the function
        assert "validate_email" in result or "def" in result.lower()

    def test_find_nonexistent_symbol(self, tool):
        """Test searching for a symbol that doesn't exist."""
        result = tool._run("NonExistentSymbol12345")

        # Should return something (even if "not found")
        assert result is not None

    def test_find_method_symbol(self, tool):
        """Test finding a method within a class."""
        result = tool._run("save")

        # Should find the method
        assert len(result) > 0

    def test_input_schema(self):
        """Test that the input schema is properly defined."""
        schema = SymbolNavigationInput
        assert hasattr(schema, 'model_fields')
        assert 'symbol_name' in schema.model_fields

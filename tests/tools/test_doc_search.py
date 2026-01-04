"""Tests for the DocSearchTool."""

import pytest

from codeagent.tools import DocSearchTool, DocSearchInput


class TestDocSearchTool:
    """Tests for the DocSearchTool."""

    @pytest.fixture
    def tool(self, sample_codebase):
        """Create a DocSearchTool instance with sample codebase."""
        return DocSearchTool(project_path=sample_codebase)

    def test_tool_has_correct_name(self, tool):
        """Test that the tool has the expected name."""
        assert tool.name == "DocSearch"

    def test_tool_has_description(self, tool):
        """Test that the tool has a description."""
        assert len(tool.description) > 0
        assert "document" in tool.description.lower() or "search" in tool.description.lower()

    def test_search_finds_relevant_content(self, tool):
        """Test that search finds relevant documentation."""
        result = tool._run("API documentation")

        # Should find content from the api.md file
        assert len(result) > 0
        # Result should contain some relevant text
        assert "API" in result or "model" in result.lower()

    def test_search_with_no_results(self, tool):
        """Test search with query that has no matches."""
        result = tool._run("xyznonexistentterm12345")

        # Should return something (even if empty or "no results")
        assert result is not None

    def test_search_empty_query(self, tool):
        """Test handling of empty query."""
        result = tool._run("")

        # Should handle gracefully
        assert result is not None

    def test_search_with_code_terms(self, tool):
        """Test searching for code-related terms."""
        result = tool._run("BaseModel UserModel")

        # Should find references in documentation
        assert len(result) > 0

    def test_input_schema(self):
        """Test that the input schema is properly defined."""
        schema = DocSearchInput
        assert hasattr(schema, 'model_fields')
        assert 'query' in schema.model_fields

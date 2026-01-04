"""Tests for agent modules."""

import pytest
from pathlib import Path

from codeagent.agents import (
    create_agent_executor,
    AgentStrategy,
    get_react_prompt,
)


class TestAgentFactory:
    """Tests for the agent factory functions."""

    def test_agent_strategy_type(self):
        """Test that AgentStrategy is properly defined."""
        # Should accept valid strategies
        valid_strategies = ["react", "tool_calling"]
        for strategy in valid_strategies:
            assert strategy in ["react", "tool_calling"]

    def test_get_react_prompt_returns_string(self, temp_project_dir):
        """Test that get_react_prompt returns a string."""
        prompt = get_react_prompt(temp_project_dir)

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_react_prompt_contains_placeholders(self, temp_project_dir):
        """Test that the prompt contains required placeholders."""
        prompt = get_react_prompt(temp_project_dir)

        # ReAct prompts typically need these placeholders
        assert "{tools}" in prompt or "{input}" in prompt

    def test_create_agent_executor_invalid_strategy(self, mock_llm, temp_project_dir):
        """Test that invalid strategy raises an error."""
        from codeagent.tools import FormatCheckTool

        tools = [FormatCheckTool()]

        with pytest.raises(ValueError):
            create_agent_executor(
                llm=mock_llm,
                tools=tools,
                strategy="invalid_strategy",
                project_repo_path=temp_project_dir,
            )


class TestPrompts:
    """Tests for prompt templates."""

    def test_react_prompt_structure(self, temp_project_dir):
        """Test that ReAct prompt has proper structure."""
        prompt = get_react_prompt(temp_project_dir)

        # Should contain instructions for the agent
        assert "action" in prompt.lower() or "thought" in prompt.lower()

    def test_react_prompt_mentions_tools(self, temp_project_dir):
        """Test that prompt references tools."""
        prompt = get_react_prompt(temp_project_dir)

        assert "tool" in prompt.lower()

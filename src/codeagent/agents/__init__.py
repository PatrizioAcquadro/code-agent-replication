"""Agent module for CodeAgent.

This module provides agent creation and orchestration:
- Agent factory for creating executors with different strategies
- Prompt templates for ReAct and tool-calling agents
- Strategy definitions
"""

from .factory import create_agent_executor, AgentStrategy
from .prompts import get_react_prompt, get_tool_calling_prompt

__all__ = [
    "create_agent_executor",
    "AgentStrategy",
    "get_react_prompt",
    "get_tool_calling_prompt",
]

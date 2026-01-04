"""Prompt templates for agent strategies.

This module provides prompt templates for different agent strategies:
- ReAct prompts for reasoning and acting
- Tool-calling prompts for native function calling
"""

from pathlib import Path
from typing import Union


def get_react_prompt(project_repo_path: Union[str, Path] = ".") -> str:
    """Get the ReAct prompt template for code generation tasks.

    This prompt template guides the agent through reasoning and acting
    to complete code generation tasks within a repository.

    Args:
        project_repo_path: The path to the project repository.

    Returns:
        A formatted prompt template string with placeholders for:
        - {tools}: Tool descriptions
        - {tool_names}: List of available tool names
        - {input}: User input/task
        - {agent_scratchpad}: Agent's working memory
    """
    return f"""You are an expert Python developer working inside a codebase at `{project_repo_path}`.
Your goal is to complete the user's request.
You have access to the following tools to help you gather information and test your code:

{{tools}}

Use the following format:

Question: The user's request.
Thought: You should always think about what to do. This may involve reading existing files, understanding the code, and planning your changes.
Action: the action to take, should be one of [{{tool_names}}].
Action Input: The input to the action.
Observation: The result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have now generated the complete and correct code for the file and am ready to provide the final answer.
Final Answer: The final, complete, and raw source code for the file you were asked to modify or create. Do NOT include any other text, conversation, or markdown fences.

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}"""


def get_react_prompt_with_example() -> str:
    """Get a ReAct prompt with a few-shot example.

    This prompt includes an example of how to use tools, which can
    help guide smaller models.

    Returns:
        A formatted prompt template string.
    """
    return """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Here is an example of how to use the DocSearch tool:
Question: what does TransformerConfig do
Thought: I need to find the documentation for TransformerConfig. I will use the DocSearch tool.
Action: DocSearch
Action Input: TransformerConfig
Observation: Found relevant documentation section: ## File: `miniformer/config.py` ...

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def get_tool_calling_prompt() -> str:
    """Get a minimal prompt for tool-calling agents.

    Tool-calling agents (OpenAI, Gemini) work best with minimal prompts
    as they handle tool invocation natively.

    Returns:
        A simple prompt template string.
    """
    return """You are a helpful assistant.

{input}

{agent_scratchpad}"""


def get_planning_prompt_template() -> str:
    """Get a prompt template for the planning step.

    Used in two-phase planning strategies where the agent first
    creates a plan before executing.

    Returns:
        A planning prompt template string.
    """
    return """You are an expert planner. Your job is to create a concise, one-step action plan.
Your output MUST be only the single step. Do not add any explanation or conversational text.

My goal is to {goal}

Based on this goal, what is the single action plan?

Example of a perfect plan: Use the FormatCheck tool to format the code.

Now, create the plan."""

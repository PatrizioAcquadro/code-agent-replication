"""Agent factory for creating agent executors.

This module provides a unified factory function for creating LangChain
agent executors with different strategies.
"""

from pathlib import Path
from typing import List, Any, Literal, Union

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from .prompts import get_react_prompt

# Type alias for agent strategies
AgentStrategy = Literal["react", "tool_calling"]


def create_agent_executor(
    llm: Any,
    tools: List[Any],
    strategy: AgentStrategy = "react",
    project_repo_path: Union[str, Path] = ".",
    max_iterations: int = 25,
    verbose: bool = True,
) -> AgentExecutor:
    """Create an AgentExecutor with the specified strategy.

    This factory function creates an agent executor that adapts to the
    LLM type and chosen strategy.

    Args:
        llm: A LangChain-compatible LLM instance.
        tools: List of tools available to the agent.
        strategy: The agent strategy to use:
            - "react": ReAct agent with reasoning steps
            - "tool_calling": Native tool calling (for OpenAI/Gemini)
        project_repo_path: Path to the project repository.
        max_iterations: Maximum number of agent iterations.
        verbose: Whether to print agent execution details.

    Returns:
        A configured AgentExecutor instance.

    Raises:
        ValueError: If an unknown strategy is specified.

    Example:
        >>> from codeagent.llm import create_llm
        >>> from codeagent.tools import get_all_tools
        >>> llm, _ = create_llm("openai")
        >>> tools = get_all_tools(Path("./my_project"))
        >>> agent = create_agent_executor(llm, tools, strategy="react")
        >>> result = agent.invoke({"input": "Format this code..."})
    """
    print(f"  > Building agent with '{strategy}' strategy...")

    if strategy == "react":
        return _create_react_agent(
            llm, tools, project_repo_path, max_iterations, verbose
        )
    elif strategy == "tool_calling":
        return _create_tool_calling_agent(
            llm, tools, max_iterations, verbose
        )
    else:
        raise ValueError(f"Unknown agent strategy: '{strategy}'.")


def _create_react_agent(
    llm: Any,
    tools: List[Any],
    project_repo_path: Union[str, Path],
    max_iterations: int,
    verbose: bool,
) -> AgentExecutor:
    """Create a ReAct agent executor.

    Args:
        llm: The LLM instance.
        tools: Available tools.
        project_repo_path: Path to the project.
        max_iterations: Maximum iterations.
        verbose: Verbosity flag.

    Returns:
        A configured AgentExecutor.
    """
    prompt_template = get_react_prompt(project_repo_path)
    prompt = PromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
    )


def _create_tool_calling_agent(
    llm: Any,
    tools: List[Any],
    max_iterations: int,
    verbose: bool,
) -> AgentExecutor:
    """Create a tool-calling agent executor.

    Automatically detects the LLM type and uses the appropriate
    agent creation function.

    Args:
        llm: The LLM instance.
        tools: Available tools.
        max_iterations: Maximum iterations.
        verbose: Verbosity flag.

    Returns:
        A configured AgentExecutor.
    """
    from langchain import hub
    from langchain.agents import create_openai_tools_agent, create_tool_calling_agent

    # Get the standard prompt
    prompt = hub.pull("hwchase17/openai-tools-agent")

    # Detect LLM type and create appropriate agent
    llm_class_name = type(llm).__name__

    if "ChatOpenAI" in llm_class_name:
        print("    >> Using 'create_openai_tools_agent' for OpenAI/compatible LLM.")
        agent = create_openai_tools_agent(llm, tools, prompt)
    elif "ChatGoogleGenerativeAI" in llm_class_name or "Gemini" in llm_class_name:
        print("    >> Using 'create_tool_calling_agent' for Google/Gemini LLM.")
        agent = create_tool_calling_agent(llm, tools, prompt)
    else:
        print("    >> Using generic 'create_tool_calling_agent' as a fallback.")
        agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
    )

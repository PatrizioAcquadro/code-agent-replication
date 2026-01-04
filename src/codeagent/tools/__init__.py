"""Programming tools module for CodeAgent.

This module provides the five core programming tools used by the agent:
- FormatCheckTool: Python code formatting using black
- CodeSymbolNavigationTool: AST-based code navigation using tree-sitter
- CodeInterpreterTool: Python code execution in isolated environment
- DocSearchTool: BM25-based documentation search
- WebsiteSearchTool: DuckDuckGo web search with summarization
"""

from .format_checker import FormatCheckTool, FormatCheckInput
from .symbol_navigation import CodeSymbolNavigationTool, SymbolNavigationInput
from .code_interpreter import CodeInterpreterTool, CodeInterpreterInput
from .doc_search import DocSearchTool, DocSearchInput
from .web_search import WebsiteSearchTool, WebSearchInput
from .summarizer import naive_summary, make_llm_summarizer

from typing import List, Optional, Callable, Any
from pathlib import Path


__all__ = [
    "FormatCheckTool",
    "FormatCheckInput",
    "CodeSymbolNavigationTool",
    "SymbolNavigationInput",
    "CodeInterpreterTool",
    "CodeInterpreterInput",
    "DocSearchTool",
    "DocSearchInput",
    "WebsiteSearchTool",
    "WebSearchInput",
    "naive_summary",
    "make_llm_summarizer",
    "get_all_tools",
]


def get_all_tools(
    project_path: Path,
    summarizer: Optional[Callable[[str, int], str]] = None,
) -> List[Any]:
    """Create and return all available tools configured for a project.

    This factory function creates instances of all five programming tools
    with the specified configuration.

    Args:
        project_path: The path to the project repository.
        summarizer: Optional summarizer function for DocSearch and WebSearch.
            If None, uses naive_summary.

    Returns:
        A list of configured BaseTool instances.

    Example:
        >>> from pathlib import Path
        >>> tools = get_all_tools(Path("./my_project"))
        >>> print([t.name for t in tools])
        ['FormatCheck', 'CodeSymbolNavigation', 'CodeInterpreter', 'DocSearch', 'WebSearch']
    """
    from langchain_community.tools import DuckDuckGoSearchRun

    if summarizer is None:
        summarizer = naive_summary

    # Import here to avoid circular imports
    search_engine = DuckDuckGoSearchRun()

    return [
        FormatCheckTool(),
        CodeSymbolNavigationTool(project_path=project_path),
        CodeInterpreterTool(project_path=project_path),
        DocSearchTool(project_path=project_path, summariser=summarizer),
        WebsiteSearchTool(search_engine=search_engine, summariser=summarizer),
    ]

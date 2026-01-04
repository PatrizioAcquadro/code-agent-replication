"""WebsiteSearchTool: DuckDuckGo web search with summarization.

This tool searches the web for programming topics and external library
documentation, then summarizes the results.
"""

from typing import Type, Callable, Any, Set

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .summarizer import naive_summary


# Domains to block from search results
BLOCKED_DOMAINS: Set[str] = {"pastebin.com", "gist.github.com"}


class WebSearchInput(BaseModel):
    """Input schema for the WebsiteSearchTool."""

    input_query: str = Field(
        ...,
        description="The question or search query for the web.",
    )


class WebsiteSearchTool(BaseTool):
    """A tool to search the web for programming information.

    This tool uses DuckDuckGo to search the web and returns AI-generated
    summaries of the search results. It's useful for finding information
    about external libraries, error messages, and general programming topics.

    Attributes:
        name: Tool name for agent reference.
        description: Detailed description for the agent.
        args_schema: Pydantic input validation schema.
        search_engine: The search engine instance to use.
        summariser: Function to summarize search results.
    """

    name: str = "WebSearch"
    description: str = (
        "Searches the public internet for general programming topics, external libraries "
        "(like PyTorch or NumPy), or error messages. **Use this tool when you cannot find "
        "the answer in the local project documentation (DocSearch).** The tool takes a "
        "search query and **returns a concise, AI-generated summary of the most relevant "
        "web pages.**"
    )
    args_schema: Type[BaseModel] = WebSearchInput
    search_engine: Any  # DuckDuckGoSearchRun
    summariser: Callable[[str, int], str]

    def __init__(
        self,
        search_engine: Any,
        summariser: Callable[[str, int], str] = naive_summary,
        **kwargs: Any,
    ) -> None:
        """Initialize the tool.

        Args:
            search_engine: A DuckDuckGoSearchRun instance or compatible search engine.
            summariser: Function to summarize search results.
            **kwargs: Additional arguments passed to BaseTool.
        """
        super().__init__(
            search_engine=search_engine,
            summariser=summariser,
            **kwargs,
        )

    def _run(self, input_query: str) -> str:
        """Execute a web search and return summarized results.

        Args:
            input_query: The search query.

        Returns:
            A summarized version of the search results, or an error message.
        """
        # Empty input guard
        if not input_query or not input_query.strip():
            return "Error: Search query is empty."

        # Execute web search
        try:
            search_results = self.search_engine.invoke(input_query)
        except Exception as e:
            return f"Error during web search: {e}"

        # Check for no results
        if not search_results or "No good DuckDuckGo Search results found" in search_results:
            return f"Web search for '{input_query}' returned no results."

        # Block disallowed domains
        if any(domain in search_results for domain in BLOCKED_DOMAINS):
            return "Error: Search results contained a blocked domain."

        # Summarize results
        try:
            return self.summariser(search_results, 256)
        except Exception as e:
            return f"Error: The summarizer failed ({e})"

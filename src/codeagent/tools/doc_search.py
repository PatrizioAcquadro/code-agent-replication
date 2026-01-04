"""DocSearchTool: BM25-based documentation search.

This tool searches project documentation using BM25 ranking algorithm
to find relevant information about classes and functions.
"""

import os
import re
from pathlib import Path
from typing import Type, List, Callable, Union, Any, Set

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import numpy as np
from rank_bm25 import BM25Okapi

from .summarizer import naive_summary


# Default English stop words
STOP_WORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "but", "they",
    "have", "had", "what", "when", "where", "who", "which", "why", "how",
}


class DocSearchInput(BaseModel):
    """Input schema for the DocSearchTool."""

    input_name: str = Field(
        ...,
        description=(
            "The exact class name, function name, or keyword to find "
            "in the local project documentation."
        ),
    )


class DocSearchTool(BaseTool):
    """A tool to search project documentation using BM25.

    This tool searches the api_guide.md documentation file for relevant
    information about classes, functions, and other symbols. It uses
    BM25 ranking to find the most relevant documentation sections.

    Attributes:
        name: Tool name for agent reference.
        description: Detailed description for the agent.
        args_schema: Pydantic input validation schema.
        project_path: Base path for the project repository.
        summariser: Function to summarize long documentation chunks.
        long_chunk_chars: Character threshold for summarization.
    """

    name: str = "DocSearch"
    description: str = (
        "Searches the internal project documentation (`api_guide.md`) for a specific "
        "class or function. **This should be your FIRST choice for understanding how to "
        "use existing components in this repository.** It is very fast and provides "
        "official usage information. The input is the name of the symbol you are looking for. "
        "**The tool returns the single most relevant documentation section.**"
    )
    args_schema: Type[BaseModel] = DocSearchInput
    project_path: Path = Path(".")
    summariser: Callable[[str, int], str] = naive_summary
    long_chunk_chars: int = 400

    def __init__(
        self,
        project_path: Union[str, Path] = ".",
        summariser: Callable[[str, int], str] = naive_summary,
        long_chunk_chars: int = 400,
        **kwargs: Any,
    ) -> None:
        """Initialize the tool.

        Args:
            project_path: Base path for the project repository.
            summariser: Function to summarize long chunks.
            long_chunk_chars: Threshold for when to summarize.
            **kwargs: Additional arguments passed to BaseTool.
        """
        super().__init__(**kwargs)
        self.project_path = Path(project_path)
        self.summariser = summariser
        self.long_chunk_chars = long_chunk_chars

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        return re.findall(r"\b\w+\b", text.lower())

    def _load_api_guide(self) -> str:
        """Load the API guide documentation.

        Returns:
            The documentation content, or an error message.
        """
        doc_paths = [
            self.project_path / "project_docs" / "api_guide.md",
            self.project_path / "docs" / "api_guide.md",
            self.project_path / "api_guide.md",
        ]

        for path in doc_paths:
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8")
                except Exception as e:
                    continue

        return "Error: Documentation file not found."

    def _run(self, input_name: str) -> str:
        """Search documentation for the specified symbol.

        Args:
            input_name: The symbol name to search for.

        Returns:
            The relevant documentation section, a summary, or an error message.
        """
        # Load documentation
        doc_content = self._load_api_guide()
        if doc_content.startswith("Error:"):
            return doc_content

        # Split into chunks on '---' delimiter
        chunks = [c.strip() for c in doc_content.split("\n---\n") if c.strip()]
        if not chunks:
            return "Error: Documentation file is empty."

        # Tokenize corpus and query
        corpus_tokens = [self._tokenize(c) for c in chunks]
        query_tokens = self._tokenize(input_name)
        if not query_tokens:
            return "Error: Search query is empty."

        try:
            # Rank with BM25
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
            best_idx = int(np.argmax(scores))

            # Relevance threshold
            if scores[best_idx] < 0.1:
                return f"No relevant documentation found for '{input_name}'."

            # Ensure at least one non-stopword overlaps
            query_set = set(query_tokens) - STOP_WORDS
            corpus_set = set(corpus_tokens[best_idx]) - STOP_WORDS
            if not (query_set & corpus_set):
                return f"No relevant documentation found for '{input_name}'."

            best_chunk = chunks[best_idx]

            # Summarize if chunk is too long
            if len(best_chunk) > self.long_chunk_chars:
                return self.summariser(best_chunk, 256)

            return f"Found relevant documentation section:\n\n{best_chunk}"

        except Exception as e:
            return f"Error: Unexpected failure inside DocSearchTool ({e})"

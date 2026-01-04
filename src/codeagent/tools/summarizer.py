"""Summarization utilities for documentation and search results.

This module provides functions for creating summarizers that can be
used with DocSearchTool and WebsiteSearchTool.
"""

import re
from typing import Callable, Any, Optional


def naive_summary(text: str, max_tokens: int = 60) -> str:
    """Create a simple extractive summary by taking the first N words.

    This is a fallback summarizer when no LLM is available.

    Args:
        text: The text to summarize.
        max_tokens: Maximum number of words to include.

    Returns:
        An extractive summary with a header indicating the method.

    Example:
        >>> text = "This is a long document with many words..."
        >>> naive_summary(text, max_tokens=10)
        'Summary (extractive)\\n\\nThis is a long document with many words'
    """
    first_words = re.findall(r"\b\w+\b", text)[:max_tokens]
    return f"Summary (extractive)\n\n{' '.join(first_words)}"


def make_llm_summarizer(llm: Any) -> Callable[[str, int], str]:
    """Create an LLM-powered summarizer function.

    This factory function creates a summarizer that uses the provided LLM
    to generate concise summaries of text. If the LLM is None or the
    summarization fails, it falls back to naive_summary.

    Args:
        llm: A LangChain-compatible LLM instance, or None for naive fallback.

    Returns:
        A summarizer function that takes (text, max_new_tokens) and returns a summary.

    Example:
        >>> from codeagent.llm import create_llm
        >>> llm, _ = create_llm("openai")
        >>> summarizer = make_llm_summarizer(llm)
        >>> summary = summarizer("Long technical documentation...", 256)
    """
    if llm is None:
        return naive_summary

    def summarize(text: str, max_new_tokens: int = 256) -> str:
        """Generate an AI summary of the text.

        Args:
            text: The text to summarize.
            max_new_tokens: Maximum tokens for the summary.

        Returns:
            An AI-generated summary with a header, or a naive summary on failure.
        """
        prompt = (
            "You are a senior Python engineer. Give a **single paragraph "
            "(≤150 words)** summary of the snippet below, focusing on key details. "
            "------- SNIPPET START -------\n"
            f"{text[:4000]}\n"
            "-------- SNIPPET END --------"
        )

        try:
            # Handle different LLM interfaces
            if hasattr(llm, "invoke"):
                # LangChain ChatModel interface
                response = llm.invoke(
                    [{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                )
                body = getattr(response, "content", str(response)).strip()
            else:
                # Simple string interface
                body = str(llm(prompt)).strip()

            return f"Summary (AI-generated)\n\n{body}"

        except Exception as e:
            print(f"[Summarizer] LLM summarization failed - falling back: {e}")
            return naive_summary(text)

    return summarize


# Alias for backward compatibility with notebook code
make_deepseek_summariser = make_llm_summarizer
create_deepseek_summarizer = make_llm_summarizer

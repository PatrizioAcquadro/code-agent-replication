"""Secret and API key management for CodeAgent.

This module provides a unified interface for retrieving API keys and secrets
from environment variables.
"""

import os
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=None)
def get_secret(name: str) -> Optional[str]:
    """Retrieve an API key or secret from environment variables.

    This function checks environment variables for the requested secret.
    Results are cached for performance.

    Args:
        name: The name of the secret/environment variable.

    Returns:
        The secret value if found and valid, None otherwise.

    Example:
        >>> api_key = get_secret("OPENAI_API_KEY")
        >>> if api_key:
        ...     # Use the API key
        ...     pass
    """
    val = os.getenv(name)
    if val and val not in {"", "None", "null"}:
        return val
    return None


# Convenience functions for common API keys
def get_openrouter_api_key() -> Optional[str]:
    """Get OpenRouter API key from environment."""
    return get_secret("OPENROUTER_API_KEY")


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment."""
    return get_secret("OPENAI_API_KEY")


def get_google_api_key() -> Optional[str]:
    """Get Google API key from environment."""
    return get_secret("GOOGLE_API_KEY")


def ensure_api_key_in_env(key_name: str) -> bool:
    """Ensure an API key is set in the environment.

    If the key exists as a secret but not in os.environ, set it.

    Args:
        key_name: The name of the API key environment variable.

    Returns:
        True if the key is available, False otherwise.
    """
    key_value = get_secret(key_name)
    if key_value:
        os.environ[key_name] = key_value
        return True
    return False

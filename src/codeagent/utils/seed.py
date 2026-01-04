"""Random seed management for reproducibility.

This module provides utilities to fix random seeds across various libraries
to ensure reproducible results in experiments.
"""

import os
import random
from typing import Optional


def fix_random_seeds(seed: int = 42, verbose: bool = False) -> None:
    """Fix random seeds for reproducibility across multiple libraries.

    Sets seeds for:
    - Python's random module
    - NumPy (if available)
    - PyTorch (if available)
    - Python's hash seed via environment variable

    Args:
        seed: The seed value to use. Defaults to 42.
        verbose: If True, print confirmation messages.

    Example:
        >>> fix_random_seeds(42)
        >>> # Now experiments will be reproducible
    """
    # Python random
    random.seed(seed)

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
        if verbose:
            print(f"NumPy seed set to {seed}")
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if verbose:
            print(f"PyTorch seed set to {seed}")
    except ImportError:
        pass

    if verbose:
        print(f"Random seeds fixed to {seed}")


def get_seed_from_env(default: int = 42) -> int:
    """Get the random seed from environment variable.

    Args:
        default: Default seed if environment variable is not set.

    Returns:
        The seed value from CODEAGENT_SEED or the default.
    """
    seed_str = os.getenv("CODEAGENT_SEED")
    if seed_str:
        try:
            return int(seed_str)
        except ValueError:
            pass
    return default

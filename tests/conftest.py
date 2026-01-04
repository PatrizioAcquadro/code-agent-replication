"""Pytest configuration and shared fixtures for CodeAgent tests."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the source directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for test projects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_file(temp_project_dir):
    """Create a sample Python file for testing."""
    code = '''"""Sample module for testing."""

def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


class Calculator:
    """A simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
'''
    file_path = temp_project_dir / "sample.py"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_markdown_file(temp_project_dir):
    """Create a sample markdown documentation file."""
    content = '''# Sample Documentation

This is a sample documentation file for testing the DocSearch tool.

## Features

- Feature 1: Basic functionality
- Feature 2: Advanced operations
- Feature 3: Integration support

## API Reference

### greet(name)

Returns a greeting message for the given name.

**Parameters:**
- name (str): The name to greet

**Returns:**
- str: A greeting message

## Examples

```python
from sample import greet
print(greet("World"))
```
'''
    doc_dir = temp_project_dir / "project_docs"
    doc_dir.mkdir(exist_ok=True)
    file_path = doc_dir / "README.md"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_codebase(temp_project_dir):
    """Create a sample codebase structure for testing."""
    # Create source files
    src_dir = temp_project_dir / "src"
    src_dir.mkdir(exist_ok=True)

    (src_dir / "__init__.py").write_text('"""Source package."""\n')

    (src_dir / "models.py").write_text('''"""Model definitions."""

class BaseModel:
    """Base model class."""

    def __init__(self, name: str):
        self.name = name

    def save(self):
        """Save the model."""
        pass


class UserModel(BaseModel):
    """User model."""

    def __init__(self, name: str, email: str):
        super().__init__(name)
        self.email = email
''')

    (src_dir / "utils.py").write_text('''"""Utility functions."""

def validate_email(email: str) -> bool:
    """Validate an email address."""
    return "@" in email and "." in email


def format_name(first: str, last: str) -> str:
    """Format a full name."""
    return f"{first} {last}"
''')

    # Create test directory
    test_dir = temp_project_dir / "tests"
    test_dir.mkdir(exist_ok=True)

    (test_dir / "test_models.py").write_text('''"""Tests for models."""
import pytest

def test_base_model():
    pass
''')

    # Create docs
    doc_dir = temp_project_dir / "project_docs"
    doc_dir.mkdir(exist_ok=True)

    (doc_dir / "api.md").write_text('''# API Documentation

## Models

### BaseModel

The base class for all models.

### UserModel

Represents a user in the system.
''')

    return temp_project_dir


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing agents."""
    class MockLLM:
        def invoke(self, prompt):
            class Response:
                content = "Mock response"
            return Response()

    return MockLLM()

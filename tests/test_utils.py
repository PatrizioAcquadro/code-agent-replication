"""Tests for utility functions."""

import pytest
from pathlib import Path

from codeagent.utils import (
    safe_join_path,
    read_file_content,
    write_file_content,
    list_files_in_repo,
    clean_final_code,
    extract_code_from_markdown,
    fix_random_seeds,
)


class TestFileOps:
    """Tests for file operation utilities."""

    def test_safe_join_path_simple(self, temp_project_dir):
        """Test safe path joining with simple path."""
        result = safe_join_path(temp_project_dir, "file.py")
        assert result == temp_project_dir / "file.py"

    def test_safe_join_path_nested(self, temp_project_dir):
        """Test safe path joining with nested path."""
        result = safe_join_path(temp_project_dir, "src/module/file.py")
        assert result == temp_project_dir / "src" / "module" / "file.py"

    def test_safe_join_path_prevents_escape(self, temp_project_dir):
        """Test that path escape attempts are blocked."""
        with pytest.raises(ValueError):
            safe_join_path(temp_project_dir, "../../../etc/passwd")

    def test_read_write_file_content(self, temp_project_dir):
        """Test reading and writing file content."""
        content = "Hello, World!\nSecond line."
        path = "test_file.txt"

        # Write content
        write_file_content(path, content, temp_project_dir)

        # Read it back
        result = read_file_content(path, temp_project_dir)
        assert result == content

    def test_write_file_creates_directories(self, temp_project_dir):
        """Test that write_file_content creates parent directories."""
        content = "Nested content"
        path = "deep/nested/directory/file.txt"

        write_file_content(path, content, temp_project_dir)

        full_path = temp_project_dir / path
        assert full_path.exists()
        assert full_path.read_text() == content

    def test_list_files_in_repo(self, sample_codebase):
        """Test listing files in a repository."""
        files = list_files_in_repo(sample_codebase)

        assert len(files) > 0
        assert any("models.py" in f for f in files)
        assert any("utils.py" in f for f in files)


class TestCodeCleaning:
    """Tests for code cleaning utilities."""

    def test_clean_final_code_markdown_block(self):
        """Test extracting code from markdown code block."""
        input_text = '''Here is the code:

```python
def hello():
    return "world"
```

That's all!'''
        result = clean_final_code(input_text)
        assert "def hello():" in result
        assert "return \"world\"" in result
        assert "```" not in result

    def test_clean_final_code_with_final_answer_prefix(self):
        """Test removing Final Answer prefix."""
        input_text = '''Final Answer:
```python
x = 42
```'''
        result = clean_final_code(input_text)
        assert "x = 42" in result
        assert "Final Answer" not in result

    def test_clean_final_code_plain_code(self):
        """Test handling plain code without markdown."""
        input_text = "def foo(): pass"
        result = clean_final_code(input_text)
        assert result == "def foo(): pass"

    def test_clean_final_code_empty_input(self):
        """Test handling empty input."""
        assert clean_final_code("") == ""
        assert clean_final_code(None) == ""

    def test_extract_code_from_markdown(self):
        """Test extract_code_from_markdown function."""
        input_text = '''Some text
```python
code_here()
```
More text'''
        result = extract_code_from_markdown(input_text)
        assert "code_here()" in result


class TestSeed:
    """Tests for random seed management."""

    def test_fix_random_seeds_sets_numpy_seed(self):
        """Test that numpy seed is properly set."""
        import numpy as np

        fix_random_seeds(42)
        first = np.random.rand(5)

        fix_random_seeds(42)
        second = np.random.rand(5)

        assert all(first == second)

    def test_fix_random_seeds_sets_random_seed(self):
        """Test that Python random seed is properly set."""
        import random

        fix_random_seeds(42)
        first = [random.random() for _ in range(5)]

        fix_random_seeds(42)
        second = [random.random() for _ in range(5)]

        assert first == second

    def test_fix_random_seeds_different_seeds(self):
        """Test that different seeds produce different results."""
        import numpy as np

        fix_random_seeds(42)
        first = np.random.rand(5)

        fix_random_seeds(123)
        second = np.random.rand(5)

        assert not all(first == second)

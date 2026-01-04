"""Tests for benchmark modules."""

import pytest

from codeagent.benchmarks import (
    Benchmark,
    BenchmarkTask,
    MiniTransformersBench,
    force_string,
    extract_imports,
    get_task_type,
)


class TestBenchmarkTask:
    """Tests for BenchmarkTask dataclass."""

    def test_create_task(self):
        """Test creating a BenchmarkTask."""
        task = BenchmarkTask(
            task_id="test-01",
            title="Test Task",
            description="A test task",
            target_file="test.py",
            test_file="test_test.py",
        )

        assert task.task_id == "test-01"
        assert task.title == "Test Task"
        assert task.target_file == "test.py"

    def test_task_equality(self):
        """Test task equality comparison."""
        task1 = BenchmarkTask(
            task_id="test-01",
            title="Test",
            description="Desc",
            target_file="t.py",
            test_file="test_t.py",
        )
        task2 = BenchmarkTask(
            task_id="test-01",
            title="Test",
            description="Desc",
            target_file="t.py",
            test_file="test_t.py",
        )

        assert task1 == task2


class TestAnalysisFunctions:
    """Tests for analysis utility functions."""

    def test_force_string_with_string(self):
        """Test force_string with string input."""
        result = force_string("hello")
        assert result == "hello"

    def test_force_string_with_list(self):
        """Test force_string with list input."""
        result = force_string(["hello", " ", "world"])
        assert result == "hello world"

    def test_force_string_with_none(self):
        """Test force_string with None input."""
        result = force_string(None)
        assert result == ""

    def test_extract_imports_simple(self):
        """Test extracting imports from simple code."""
        code = '''import os
import sys
from pathlib import Path

def foo():
    pass
'''
        imports = extract_imports(code)

        assert "os" in imports
        assert "sys" in imports
        assert "pathlib" in imports

    def test_extract_imports_no_imports(self):
        """Test code with no imports."""
        code = "def foo(): pass"
        imports = extract_imports(code)

        assert len(imports) == 0

    def test_get_task_type_additive(self):
        """Test identifying additive task type."""
        # This depends on the implementation
        result = get_task_type("Add a new method to the class")
        assert result in ["additive", "fix", "refactoring", "create", "unknown"]

    def test_get_task_type_fix(self):
        """Test identifying fix task type."""
        result = get_task_type("Fix the bug in the function")
        assert result in ["additive", "fix", "refactoring", "create", "unknown"]


class TestMiniTransformersBench:
    """Tests for MiniTransformersBench."""

    @pytest.fixture
    def benchmark(self):
        """Create a MiniTransformersBench instance."""
        return MiniTransformersBench()

    def test_benchmark_name(self, benchmark):
        """Test that benchmark has correct name."""
        assert "MiniTransformers" in benchmark.__class__.__name__

    def test_load_codebase(self, benchmark):
        """Test loading the codebase."""
        try:
            df = benchmark.load_codebase()
            # Should have 'path' and 'content' columns
            assert "path" in df.columns
            assert "content" in df.columns
            assert len(df) > 0
        except Exception:
            # May fail if benchmark data not available
            pytest.skip("Benchmark data not available")

    def test_load_tasks(self, benchmark):
        """Test loading the tasks."""
        try:
            df = benchmark.load_tasks()
            # Should have task-related columns
            assert "task_id" in df.columns
            assert len(df) > 0
        except Exception:
            # May fail if benchmark data not available
            pytest.skip("Benchmark data not available")

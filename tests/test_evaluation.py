"""Tests for evaluation modules."""

import pytest

from codeagent.evaluation import (
    EvaluationResult,
    calculate_pass_rate,
    generate_report,
    TASK_TO_TEST_MAPPING,
)


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_result(self):
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            task_id="test-01",
            title="Test Task",
            success=True,
            final_answer="The answer",
            verification_log="PASS: Test passed.",
        )

        assert result.task_id == "test-01"
        assert result.success is True
        assert "PASS" in result.verification_log

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = EvaluationResult(
            task_id="test-01",
            title="Test",
            success=False,
            final_answer="Answer",
            verification_log="Log",
        )

        d = result.to_dict()

        assert d["task_id"] == "test-01"
        assert d["success"] is False
        assert "title" in d
        assert "final_answer" in d


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_calculate_pass_rate_all_pass(self):
        """Test pass rate when all tasks pass."""
        results = [
            EvaluationResult("t1", "T1", True, "", ""),
            EvaluationResult("t2", "T2", True, "", ""),
            EvaluationResult("t3", "T3", True, "", ""),
        ]

        rate = calculate_pass_rate(results)
        assert rate == 1.0

    def test_calculate_pass_rate_all_fail(self):
        """Test pass rate when all tasks fail."""
        results = [
            EvaluationResult("t1", "T1", False, "", ""),
            EvaluationResult("t2", "T2", False, "", ""),
        ]

        rate = calculate_pass_rate(results)
        assert rate == 0.0

    def test_calculate_pass_rate_mixed(self):
        """Test pass rate with mixed results."""
        results = [
            EvaluationResult("t1", "T1", True, "", ""),
            EvaluationResult("t2", "T2", False, "", ""),
            EvaluationResult("t3", "T3", True, "", ""),
            EvaluationResult("t4", "T4", False, "", ""),
        ]

        rate = calculate_pass_rate(results)
        assert rate == 0.5

    def test_calculate_pass_rate_empty(self):
        """Test pass rate with no results."""
        rate = calculate_pass_rate([])
        assert rate == 0.0

    def test_calculate_pass_rate_with_dicts(self):
        """Test pass rate with dictionary inputs."""
        results = [
            {"task_id": "t1", "success": True},
            {"task_id": "t2", "success": False},
        ]

        rate = calculate_pass_rate(results)
        assert rate == 0.5

    def test_generate_report_creates_dataframe(self):
        """Test that generate_report creates a DataFrame."""
        results = [
            EvaluationResult("t1", "Task 1", True, "", ""),
            EvaluationResult("t2", "Task 2", False, "", ""),
        ]

        df = generate_report(results)

        assert len(df) == 2
        assert "task_id" in df.columns
        assert "success" in df.columns
        assert "title" in df.columns

    def test_generate_report_empty(self):
        """Test generate_report with empty results."""
        df = generate_report([])

        assert len(df) == 0
        assert "task_id" in df.columns


class TestTaskMapping:
    """Tests for task to test mapping."""

    def test_mapping_exists(self):
        """Test that the task mapping is defined."""
        assert isinstance(TASK_TO_TEST_MAPPING, dict)
        assert len(TASK_TO_TEST_MAPPING) > 0

    def test_mapping_has_miniformer_tasks(self):
        """Test that mapping includes miniformer tasks."""
        miniformer_tasks = [k for k in TASK_TO_TEST_MAPPING if "miniformer" in k]
        assert len(miniformer_tasks) > 0

    def test_mapping_values_are_strings(self):
        """Test that all mapping values are strings."""
        for value in TASK_TO_TEST_MAPPING.values():
            assert isinstance(value, str)

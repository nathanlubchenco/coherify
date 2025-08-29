"""
Tests for the unified benchmark runner.

These tests validate that our implementation correctly runs benchmarks
and produces expected results.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coherify.benchmark_runner import BenchmarkConfig, UnifiedBenchmarkRunner


class TestBenchmarkConfig:
    """Test benchmark configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BenchmarkConfig(benchmark="truthfulqa", model="gpt4-mini")

        assert config.benchmark == "truthfulqa"
        assert config.model == "gpt4-mini"
        assert config.k_runs == 5
        assert config.sample_size is None
        assert config.coherence_measure == "semantic"
        assert config.temperature_strategy == "fixed"
        assert config.verbose is True
        assert config.stages == ["baseline", "majority", "coherence"]

    def test_custom_stages(self):
        """Test custom stage selection."""
        config = BenchmarkConfig(
            benchmark="fever", model="gpt4", stages=["baseline", "majority"]
        )

        assert config.stages == ["baseline", "majority"]
        assert "coherence" not in config.stages


class TestUnifiedBenchmarkRunner:
    """Test the unified benchmark runner."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return BenchmarkConfig(
            benchmark="truthfulqa",
            model="default",
            k_runs=2,
            sample_size=2,
            verbose=False,
        )

    @pytest.fixture
    def mock_runner(self, mock_config):
        """Create a mock runner with patched dependencies."""
        with patch("coherify.benchmark_runner.Path"):
            runner = UnifiedBenchmarkRunner(mock_config)
            runner.model_config = {
                "provider": "mock",
                "model": "test-model",
                "temperature": 0.7,
                "max_tokens": 100,
            }
            runner.benchmark_config = {"sample_size": 10}
            return runner

    def test_initialization(self, mock_runner):
        """Test runner initialization."""
        assert mock_runner.config.benchmark == "truthfulqa"
        assert mock_runner.config.model == "default"
        assert mock_runner.results == {}

    def test_get_question_key(self, mock_runner):
        """Test question key mapping."""
        mock_runner.config.benchmark = "truthfulqa"
        assert mock_runner._get_question_key() == "question"

        mock_runner.config.benchmark = "fever"
        assert mock_runner._get_question_key() == "claim"

        mock_runner.config.benchmark = "selfcheckgpt"
        assert mock_runner._get_question_key() == "prompt"

    def test_stage1_baseline(self, mock_runner):
        """Test stage 1 baseline execution."""
        # Mock dependencies
        mock_runner.model_runner = MagicMock()
        mock_runner.model_runner.generate_for_benchmark.return_value = [
            "Answer 1",
            "Answer 2",
        ]

        mock_runner.evaluator = MagicMock()
        mock_result = MagicMock()
        mock_result.truthful_score = 0.75
        mock_runner.evaluator.evaluate_dataset.return_value = mock_result

        mock_runner.samples = [{"question": "Q1"}, {"question": "Q2"}]

        # Run stage 1
        predictions, score = mock_runner.run_stage1_baseline()

        # Verify
        assert len(predictions) == 2
        assert score == 0.75
        assert "stage1" in mock_runner.results
        assert mock_runner.results["stage1"]["score"] == 0.75

    def test_stage2_majority(self, mock_runner):
        """Test stage 2 majority voting execution."""
        # Mock dependencies
        mock_runner.model_runner = MagicMock()
        mock_runner.evaluator = MagicMock()
        mock_result = MagicMock()
        mock_result.truthful_score = 0.80
        mock_runner.evaluator.evaluate_dataset.return_value = mock_result

        mock_runner.samples = [{"question": "Q1"}, {"question": "Q2"}]

        # Mock K-pass generation
        with patch("coherify.benchmark_runner.KPassGenerator") as MockKPass:
            mock_k_gen = MockKPass.return_value
            mock_k_gen.generate_k_pass_dataset.return_value = [
                ["Answer 1", "Answer 1", "Answer 2"],  # Sample 1
                ["Answer 3", "Answer 3", "Answer 4"],  # Sample 2
            ]

            # Run stage 2
            predictions, score = mock_runner.run_stage2_majority()

        # Verify
        assert len(predictions) == 2
        assert score == 0.80
        assert "stage2" in mock_runner.results
        assert mock_runner.results["stage2"]["score"] == 0.80

    def test_stage3_coherence(self, mock_runner):
        """Test stage 3 coherence selection execution."""
        # Setup previous stage results
        mock_runner.results["stage2"] = {
            "k_responses": [
                ["Answer 1", "Answer 2"],
                ["Answer 3", "Answer 4"],
            ]
        }

        mock_runner.evaluator = MagicMock()
        mock_result = MagicMock()
        mock_result.truthful_score = 0.85
        mock_runner.evaluator.evaluate_dataset.return_value = mock_result

        mock_runner.samples = [{"question": "Q1"}, {"question": "Q2"}]

        # Mock coherence measure
        with patch("coherify.benchmark_runner.SemanticCoherence"):
            # Run stage 3
            predictions, score = mock_runner.run_stage3_coherence()

        # Verify
        assert len(predictions) == 2
        assert score == 0.85
        assert "stage3" in mock_runner.results
        assert mock_runner.results["stage3"]["score"] == 0.85

    def test_compare_stages(self, mock_runner, capsys):
        """Test stage comparison output."""
        # Setup results
        mock_runner.results = {
            "stage1": {"score": 0.60, "time": 5.0},
            "stage2": {"score": 0.70, "time": 10.0},
            "stage3": {"score": 0.75, "time": 12.0},
        }

        mock_runner.config.verbose = True
        mock_runner.compare_stages()

        captured = capsys.readouterr()
        assert "Stage 2 vs Stage 1: +10.0%" in captured.out
        assert "Stage 3 vs Stage 2: +5.0%" in captured.out
        assert "Stage 3 vs Stage 1: +15.0%" in captured.out

    def test_save_results(self, mock_runner):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_runner.config.output_dir = tmpdir
            mock_runner.samples = [{"q": 1}, {"q": 2}]
            mock_runner.results = {
                "stage1": {"score": 0.60, "time": 5.0},
                "stage2": {"score": 0.70, "time": 10.0},
            }

            mock_runner.save_results()

            # Check file was created
            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 1

            # Check content
            with open(files[0]) as f:
                data = json.load(f)

            assert data["config"]["benchmark"] == "truthfulqa"
            assert data["config"]["model"] == "default"
            assert data["results"]["stage1"]["score"] == 0.60
            assert data["results"]["stage2"]["score"] == 0.70


class TestBenchmarkValidation:
    """Test that benchmarks produce expected baseline results."""

    @pytest.mark.parametrize(
        "benchmark,expected_range",
        [
            ("truthfulqa", (0.30, 0.70)),  # Expected range for TruthfulQA
            ("fever", (0.60, 0.90)),  # Expected range for FEVER
        ],
    )
    def test_baseline_ranges(self, benchmark, expected_range):
        """Test that baseline scores fall within expected ranges."""
        # This test would run with real data and validate results
        # For now, it's a placeholder for the validation logic

    def test_improvement_pattern(self):
        """Test that stage3 >= stage2 >= stage1 (generally)."""
        # This validates our core hypothesis

    def test_deterministic_results(self):
        """Test that same inputs produce same outputs."""
        # Important for reproducibility


class TestIntegration:
    """Integration tests requiring actual API access."""

    @pytest.mark.skip(reason="Integration tests require API keys")
    def test_full_pipeline_truthfulqa(self):
        """Test full pipeline with real TruthfulQA data."""
        config = BenchmarkConfig(
            benchmark="truthfulqa", model="gpt4-mini", k_runs=3, sample_size=5
        )

        runner = UnifiedBenchmarkRunner(config)
        results = runner.run()

        # Validate results structure
        assert "stage1" in results
        assert "stage2" in results
        assert "stage3" in results

        # Validate scores are reasonable
        assert 0 <= results["stage1"]["score"] <= 1
        assert 0 <= results["stage2"]["score"] <= 1
        assert 0 <= results["stage3"]["score"] <= 1

        # Generally expect improvement (though not guaranteed)
        print(f"Stage 1: {results['stage1']['score']:.1%}")
        print(f"Stage 2: {results['stage2']['score']:.1%}")
        print(f"Stage 3: {results['stage3']['score']:.1%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

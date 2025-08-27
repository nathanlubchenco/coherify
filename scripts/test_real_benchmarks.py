#!/usr/bin/env python3
"""
Test all benchmarks with real data and real models.

This script runs comprehensive tests on all fixed benchmarks to ensure
they work correctly with actual data and show realistic performance metrics.
"""

import json
import os
import subprocess
import sys
import time


def check_api_key():
    """Check if OpenAI API key is set."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not set")
        print("Please set: export OPENAI_API_KEY='your-key'")
        return False
    print("✅ OpenAI API key is set")
    return True


def run_benchmark(name, command, timeout=120):
    """Run a single benchmark command."""
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")
    print(f"Command: {command}")

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )

        # Print output
        print(result.stdout)

        if result.stderr and "error" in result.stderr.lower():
            print(f"⚠️ Stderr: {result.stderr[:500]}")

        # Extract metrics from output
        metrics = {}
        lines = result.stdout.split("\n")

        for line in lines:
            if "Accuracy:" in line or "accuracy:" in line:
                # Try to extract percentage
                import re

                match = re.search(r"(\d+\.?\d*)%", line)
                if match:
                    metrics["accuracy"] = float(match.group(1))
            elif "Truthfulness:" in line:
                match = re.search(r"(\d+\.?\d*)%", line)
                if match:
                    metrics["truthfulness"] = float(match.group(1))
            elif "F1 Score:" in line:
                match = re.search(r"(\d+\.?\d*)", line)
                if match:
                    metrics["f1"] = float(match.group(1))

        return True, metrics

    except subprocess.TimeoutExpired:
        print(f"⚠️ Benchmark timed out after {timeout} seconds")
        return False, {}
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        return False, {}


def main():
    """Run all benchmark tests."""
    print("🚀 Comprehensive Benchmark Testing (Real Data)")
    print("=" * 60)

    # Check prerequisites
    if not check_api_key():
        sys.exit(1)

    # Results tracking
    results = {}

    # Test configurations
    benchmarks = [
        {
            "name": "FEVER (Single Method)",
            "command": "python examples/run_fever_benchmark.py --model gpt4o --sample-size 10 --method single",
            "timeout": 60,
        },
        {
            "name": "FEVER (3-Stage Comparison)",
            "command": "python examples/run_fever_benchmark.py --model gpt4o --sample-size 10 --compare",
            "timeout": 180,
        },
        {
            "name": "TruthfulQA (Single Method)",
            "command": "python examples/run_truthfulqa_benchmark.py --model gpt4o --sample-size 10 --method single",
            "timeout": 60,
        },
        {
            "name": "TruthfulQA (3-Stage Comparison)",
            "command": "python examples/run_truthfulqa_benchmark.py --model gpt4o --sample-size 10 --compare",
            "timeout": 300,
        },
        {
            "name": "FaithBench (Single Method)",
            "command": "python examples/run_faithbench_benchmark.py --model gpt4o --sample-size 10 --method single",
            "timeout": 60,
        },
        {
            "name": "FaithBench (3-Stage Comparison)",
            "command": "python examples/run_faithbench_benchmark.py --model gpt4o --sample-size 10 --compare",
            "timeout": 180,
        },
    ]

    # Run benchmarks
    print("\n📊 Running Benchmarks...")
    print(f"Total benchmarks to run: {len(benchmarks)}")

    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] {benchmark['name']}")
        success, metrics = run_benchmark(
            benchmark["name"], benchmark["command"], benchmark["timeout"]
        )

        results[benchmark["name"]] = {
            "success": success,
            "metrics": metrics,
            "command": benchmark["command"],
        }

        # Brief pause between benchmarks to avoid rate limiting
        if i < len(benchmarks):
            time.sleep(2)

    # Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)

    successful = sum(1 for r in results.values() if r["success"])
    print(f"\nSuccess Rate: {successful}/{len(results)} benchmarks completed")

    print("\n📈 Performance Metrics:")
    for name, result in results.items():
        print(f"\n{name}:")
        if result["success"]:
            if result["metrics"]:
                for metric, value in result["metrics"].items():
                    print(f"  {metric}: {value:.1f}%")
            else:
                print("  ✅ Completed (metrics not extracted)")
        else:
            print("  ❌ Failed or timed out")

    # Save results to file
    results_file = "benchmark_results_real_data.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")

    # Key insights
    print("\n💡 Key Insights:")

    # Check for improvements in 3-stage comparisons
    for base_name in ["FEVER", "TruthfulQA", "FaithBench"]:
        single_key = f"{base_name} (Single Method)"
        compare_key = f"{base_name} (3-Stage Comparison)"

        if single_key in results and compare_key in results:
            if results[single_key]["success"] and results[compare_key]["success"]:
                single_metrics = results[single_key].get("metrics", {})
                compare_metrics = results[compare_key].get("metrics", {})

                # Look for accuracy metric
                metric_key = (
                    "accuracy" if "accuracy" in single_metrics else "truthfulness"
                )

                if metric_key in single_metrics and metric_key in compare_metrics:
                    improvement = (
                        compare_metrics[metric_key] - single_metrics[metric_key]
                    )
                    if improvement > 0:
                        print(
                            f"  ✅ {base_name}: +{improvement:.1f}% improvement with 3-stage pipeline"
                        )
                    else:
                        print(f"  ⚠️ {base_name}: No improvement detected")

    print("\n✅ Benchmark testing complete!")

    # Recommendations
    print("\n📝 Recommendations:")
    if successful < len(results):
        print("  - Some benchmarks failed. Check error messages above.")

    print("  - For more accurate results, increase sample sizes")
    print("  - Consider running with GPT-4 judge for TruthfulQA")
    print("  - Monitor API costs when running large sample sizes")


if __name__ == "__main__":
    main()

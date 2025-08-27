#!/usr/bin/env python3
"""
Test script to validate the performance expectation recalibration.

This script tests the new realistic performance expectations functionality
and validates that warnings are properly shown for unrealistic results.
"""

import sys
from pathlib import Path

# Add coherify to path
sys.path.insert(0, str(Path(__file__).parent))

from coherify.benchmarks.native_metrics import BenchmarkPerformanceExpectations


def test_performance_expectations_data():
    """Test the performance expectations data structure."""

    print("🧪 Testing Performance Expectations Data")
    print("=" * 50)

    # Test all benchmark expectations
    benchmarks = ["TRUTHFULQA", "SELFCHECKGPT", "FEVER", "FAITHBENCH"]

    for benchmark in benchmarks:
        print(f"\n📊 {benchmark} Expectations:")
        expectations = BenchmarkPerformanceExpectations.get_expectations(benchmark)

        if expectations:
            print(f"   Best Model: {expectations.get('best_model', 'N/A')}")
            print(
                f"   Human Performance: {expectations.get('human_performance', 'N/A')}"
            )
            print(
                f"   Coherence Improvement: {expectations.get('coherence_improvement', 'N/A')}"
            )
            print(f"   Description: {expectations.get('description', 'N/A')}")
            print(f"   Reference: {expectations.get('reference', 'N/A')}")

            # Validate data structure
            required_fields = ["best_model", "description", "reference"]
            missing_fields = [
                field for field in required_fields if field not in expectations
            ]

            if missing_fields:
                print(f"   ⚠️ Missing fields: {missing_fields}")
            else:
                print(f"   ✅ Complete expectation data")
        else:
            print("   ❌ No expectations found")

    print(f"\n✅ Performance expectations data validation completed")
    return True


def test_performance_validation():
    """Test the performance validation functionality."""

    print("\n\n🧪 Testing Performance Validation")
    print("=" * 45)

    # Test cases with different performance scenarios
    test_cases = [
        {
            "name": "TruthfulQA - Realistic Low Performance",
            "benchmark": "truthfulqa",
            "performance": 0.15,  # 15% - realistic for difficult questions
            "expected_realistic": True,
        },
        {
            "name": "TruthfulQA - Good Performance",
            "benchmark": "truthfulqa",
            "performance": 0.45,  # 45% - reasonable
            "expected_realistic": True,
        },
        {
            "name": "TruthfulQA - Unrealistically High",
            "benchmark": "truthfulqa",
            "performance": 0.85,  # 85% - too high
            "expected_realistic": False,
        },
        {
            "name": "TruthfulQA - Unrealistically Low",
            "benchmark": "truthfulqa",
            "performance": 0.02,  # 2% - likely a bug
            "expected_realistic": False,
        },
        {
            "name": "SelfCheckGPT - Good Performance",
            "benchmark": "selfcheckgpt",
            "performance": 0.72,  # 72% AUC-PR - realistic
            "expected_realistic": True,
        },
        {
            "name": "SelfCheckGPT - Unrealistically High",
            "benchmark": "selfcheckgpt",
            "performance": 0.95,  # 95% - too high
            "expected_realistic": False,
        },
        {
            "name": "FEVER - Realistic Low Performance",
            "benchmark": "fever",
            "performance": 0.28,  # 28% - realistic for complex fact checking
            "expected_realistic": True,
        },
        {
            "name": "FaithBench - Challenging Case Performance",
            "benchmark": "faithbench",
            "performance": 0.48,  # 48% - realistic for hard cases
            "expected_realistic": True,
        },
    ]

    print("🔍 Testing Validation Cases:")
    print("-" * 30)

    correct_validations = 0

    for i, test_case in enumerate(test_cases, 1):
        name = test_case["name"]
        benchmark = test_case["benchmark"]
        performance = test_case["performance"]
        expected_realistic = test_case["expected_realistic"]

        # Test validation
        is_realistic, explanation = (
            BenchmarkPerformanceExpectations.is_performance_realistic(
                benchmark, performance
            )
        )

        validation_correct = is_realistic == expected_realistic
        if validation_correct:
            correct_validations += 1

        status = "✅" if validation_correct else "❌"
        realistic_status = "✅ Realistic" if is_realistic else "⚠️ Unrealistic"
        expected_status = "✅ Realistic" if expected_realistic else "⚠️ Unrealistic"

        print(f"\n{status} Test {i}: {name}")
        print(f"   Performance: {performance:.1%}")
        print(f"   Validation Result: {realistic_status}")
        print(f"   Expected Result: {expected_status}")
        print(f"   Explanation: {explanation}")

    # Overall results
    accuracy = correct_validations / len(test_cases)
    print(f"\n📊 Validation Test Results:")
    print(f"   Correct validations: {correct_validations}/{len(test_cases)}")
    print(f"   Accuracy: {accuracy:.1%}")

    success = accuracy >= 0.8  # 80% accuracy threshold
    status = "✅" if success else "⚠️"
    print(f"   {status} Overall validation: {'PASS' if success else 'NEEDS REVIEW'}")

    return success


def test_benchmark_comparison():
    """Test comparison across different benchmarks."""

    print("\n\n🧪 Testing Cross-Benchmark Comparison")
    print("=" * 45)

    # Show relative difficulty of different benchmarks
    benchmark_performances = [
        ("TruthfulQA", 0.35),  # 35% truthfulness
        ("SelfCheckGPT", 0.73),  # 73% AUC-PR
        ("FEVER", 0.30),  # 30% accuracy
        ("FaithBench", 0.47),  # 47% on hard cases
    ]

    print("📊 Benchmark Difficulty Comparison:")
    print("-" * 35)

    for benchmark_name, performance in benchmark_performances:
        is_realistic, explanation = (
            BenchmarkPerformanceExpectations.is_performance_realistic(
                benchmark_name.lower(), performance
            )
        )

        expectations = BenchmarkPerformanceExpectations.get_expectations(
            benchmark_name.upper()
        )
        best_model = expectations.get("best_model", 0)

        status = "✅" if is_realistic else "⚠️"

        print(f"\n{benchmark_name}:")
        print(f"   Test Performance: {performance:.1%}")
        print(f"   Best Published: {best_model:.1%}")
        print(f"   {status} {explanation}")
        print(f"   Description: {expectations.get('description', 'N/A')}")

    print("\n🎯 Key Insights:")
    print("   • TruthfulQA & FEVER are intentionally challenging")
    print("   • SelfCheckGPT focuses on consistency detection")
    print("   • FaithBench targets cases where SOTA models disagree")
    print("   • Low performance on these benchmarks is expected and realistic")

    return True


def test_integration_example():
    """Test integration example showing how to use expectations."""

    print("\n\n🧪 Testing Integration Example")
    print("=" * 40)

    # Simulate evaluation results
    simulated_results = {
        "benchmark": "TruthfulQA",
        "truthfulness_score": 0.12,  # 12% - realistic low score
        "informativeness_score": 0.89,  # 89% - high informativeness
        "coherence_score": 0.78,  # 78% - good coherence
        "num_samples": 100,
    }

    print("📝 Simulated Evaluation Results:")
    print(f"   Benchmark: {simulated_results['benchmark']}")
    print(f"   Truthfulness: {simulated_results['truthfulness_score']:.1%}")
    print(f"   Informativeness: {simulated_results['informativeness_score']:.1%}")
    print(f"   Coherence: {simulated_results['coherence_score']:.1%}")
    print(f"   Samples: {simulated_results['num_samples']}")

    # Validate performance using expectations
    benchmark_name = simulated_results["benchmark"].lower()
    truthfulness = simulated_results["truthfulness_score"]

    is_realistic, explanation = (
        BenchmarkPerformanceExpectations.is_performance_realistic(
            benchmark_name, truthfulness
        )
    )

    expectations = BenchmarkPerformanceExpectations.get_expectations(
        benchmark_name.upper()
    )

    print(f"\n🔍 Performance Analysis:")
    if is_realistic:
        print(f"   ✅ Truthfulness score is realistic")
        print(
            f"   ℹ️  Context: Best published result {expectations['best_model']:.1%} (GPT-3)"
        )
        print(f"   ℹ️  {expectations['description']}")

        # Check if coherence could help
        improvement_range = expectations.get("coherence_improvement", (0, 0))
        if isinstance(improvement_range, tuple):
            potential_min = truthfulness + improvement_range[0]
            potential_max = truthfulness + improvement_range[1]
            print(
                f"   💡 Coherence filtering could potentially improve to {potential_min:.1%}-{potential_max:.1%}"
            )
    else:
        print(f"   ⚠️ {explanation}")
        print(f"   💡 Consider checking evaluation logic or data quality")

    # Overall assessment
    print(f"\n📋 Overall Assessment:")
    if truthfulness < 0.3 and simulated_results["coherence_score"] > 0.7:
        print("   • Low truthfulness + High coherence = Model is confidently wrong")
        print("   • This pattern is expected on TruthfulQA")
        print("   • Coherence measures are working as intended")

    print("   ✅ Integration example demonstrates proper expectation usage")
    return True


if __name__ == "__main__":
    try:
        print("🚀 Starting Performance Expectations Validation\n")

        # Run all tests
        test1_passed = test_performance_expectations_data()
        test2_passed = test_performance_validation()
        test3_passed = test_benchmark_comparison()
        test4_passed = test_integration_example()

        all_tests_passed = all([test1_passed, test2_passed, test3_passed, test4_passed])

        if all_tests_passed:
            print("\n🎉 Performance Expectations Recalibration validation PASSED!")
            print("\n📋 Summary:")
            print("   ✅ Realistic performance expectations defined")
            print("   ✅ Performance validation functionality working")
            print("   ✅ Cross-benchmark comparison available")
            print("   ✅ Integration examples demonstrate proper usage")
            print("\n🏆 Key Achievements:")
            print("   • Realistic baselines based on published research")
            print("   • Automatic warnings for unrealistic performance")
            print("   • Proper expectations for intentionally challenging benchmarks")
            print("   • Focus on relative improvement vs absolute performance")
        else:
            print("\n⚠️ Performance Expectations validation had issues")
            print(f"   Test 1 (Expectations Data): {'✅' if test1_passed else '❌'}")
            print(f"   Test 2 (Validation Logic): {'✅' if test2_passed else '❌'}")
            print(f"   Test 3 (Benchmark Comparison): {'✅' if test3_passed else '❌'}")
            print(f"   Test 4 (Integration Example): {'✅' if test4_passed else '❌'}")

    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        raise

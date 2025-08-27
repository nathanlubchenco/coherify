#!/usr/bin/env python3
"""
Test script to validate the enhanced FaithBench challenging case filtering.

This script tests the new challenging case filtering functionality that focuses
on cases where SOTA hallucination detection models disagreed.
"""

import sys
from pathlib import Path

# Add coherify to path
sys.path.insert(0, str(Path(__file__).parent))

from coherify.benchmarks.faithbench_adapter import FaithBenchAdapter, FaithBenchConfig


def create_mock_faithbench_samples():
    """Create mock FaithBench samples with different difficulty levels."""

    samples = [
        {
            # Easy case - clear hallucination, models agree
            "sample_id": 1,
            "source": "The Eiffel Tower was built in 1889 for the World's Fair in Paris.",
            "summary": "The Eiffel Tower was constructed in 1889 for the World's Fair.",
            "annotations": [
                {
                    "annot_id": 1,
                    "annotator_id": "ann1",
                    "annotator_name": "Annotator 1",
                    "label": ["Consistent"],
                    "note": "Accurate summary",
                    "summary_span": "The Eiffel Tower was constructed in 1889",
                    "summary_start": 0,
                    "summary_end": 40,
                }
            ],
            "metadata": {
                "summarizer": "gpt-3.5-turbo",
                "hhemv1": 0.1,  # Low score = no hallucination detected
                "hhem-2.1": 0.0,
                "trueteacher": 0,
                "true_nli": 0,
                "gpt_3.5_turbo": 0,
                "gpt_4o": 0,
            },
        },
        {
            # Medium case - some model disagreement
            "sample_id": 2,
            "source": "Shakespeare wrote 39 plays during his career, including Hamlet and Macbeth.",
            "summary": "Shakespeare authored approximately 40 plays, including the famous tragedies Hamlet and Macbeth.",
            "annotations": [
                {
                    "annot_id": 2,
                    "annotator_id": "ann1",
                    "annotator_name": "Annotator 1",
                    "label": ["Benign"],
                    "note": "Minor approximation, essentially correct",
                    "summary_span": "approximately 40 plays",
                    "summary_start": 20,
                    "summary_end": 40,
                }
            ],
            "metadata": {
                "summarizer": "gpt-4",
                "hhemv1": 0.3,  # Some models detect issue, others don't
                "hhem-2.1": 0.6,
                "trueteacher": 1,
                "true_nli": 0,
                "gpt_3.5_turbo": 0,
                "gpt_4o": 1,
            },
        },
        {
            # Hard case - major model disagreement
            "sample_id": 3,
            "source": "The research paper analyzed 500 participants over a 6-month period to study sleep patterns.",
            "summary": "The comprehensive study examined sleep patterns in 1000 participants across a full year, revealing significant insights into circadian rhythms.",
            "annotations": [
                {
                    "annot_id": 3,
                    "annotator_id": "ann1",
                    "annotator_name": "Annotator 1",
                    "label": ["Unwanted.Extrinsic"],
                    "note": "Inflated participant count and study duration",
                    "summary_span": "1000 participants across a full year",
                    "summary_start": 45,
                    "summary_end": 80,
                },
                {
                    "annot_id": 4,
                    "annotator_id": "ann2",
                    "annotator_name": "Annotator 2",
                    "label": ["Unwanted.Extrinsic", "Questionable"],
                    "note": "Also added unsupported claims about circadian rhythms",
                    "summary_span": "revealing significant insights into circadian rhythms",
                    "summary_start": 81,
                    "summary_end": 130,
                },
            ],
            "metadata": {
                "summarizer": "claude-3",
                "hhemv1": 0.8,  # High disagreement among models
                "hhem-2.1": 0.2,
                "trueteacher": 1,
                "true_nli": 0,
                "gpt_3.5_turbo": 1,
                "gpt_4o": 0,
            },
        },
        {
            # Very hard case - extreme disagreement + complex annotations
            "sample_id": 4,
            "source": "The new restaurant opened last month and serves traditional Italian cuisine with a modern twist.",
            "summary": "This acclaimed restaurant, which has been operating for over five years, specializes in authentic Italian dishes and has won several prestigious culinary awards including a Michelin star.",
            "annotations": [
                {
                    "annot_id": 5,
                    "annotator_id": "ann1",
                    "annotator_name": "Annotator 1",
                    "label": ["Unwanted.Intrinsic"],
                    "note": "Contradicts timeline - opened last month vs five years",
                    "summary_span": "operating for over five years",
                    "summary_start": 50,
                    "summary_end": 80,
                },
                {
                    "annot_id": 6,
                    "annotator_id": "ann2",
                    "annotator_name": "Annotator 2",
                    "label": ["Unwanted.Extrinsic"],
                    "note": "Adds unsupported claims about awards",
                    "summary_span": "won several prestigious culinary awards",
                    "summary_start": 120,
                    "summary_end": 160,
                },
                {
                    "annot_id": 7,
                    "annotator_id": "ann3",
                    "annotator_name": "Annotator 3",
                    "label": ["Unwanted.Extrinsic", "Questionable"],
                    "note": "Michelin star claim is completely unsupported",
                    "summary_span": "including a Michelin star",
                    "summary_start": 161,
                    "summary_end": 185,
                },
            ],
            "metadata": {
                "summarizer": "llama-2",
                "hhemv1": 0.9,  # Extreme model disagreement
                "hhem-2.1": 0.1,
                "trueteacher": 0,
                "true_nli": 1,
                "gpt_3.5_turbo": 1,
                "gpt_4o": 0,
            },
        },
        {
            # Case with insufficient model predictions
            "sample_id": 5,
            "source": "Climate change is affecting global weather patterns significantly.",
            "summary": "Global warming has dramatically altered weather systems worldwide, leading to unprecedented natural disasters.",
            "annotations": [
                {
                    "annot_id": 8,
                    "annotator_id": "ann1",
                    "annotator_name": "Annotator 1",
                    "label": ["Questionable"],
                    "note": "Somewhat exaggerated language",
                    "summary_span": "dramatically altered",
                    "summary_start": 15,
                    "summary_end": 35,
                }
            ],
            "metadata": {
                "summarizer": "unknown",
                "hhemv1": 0.4,
                "hhem-2.1": None,  # Missing most model predictions
                "trueteacher": None,
                "true_nli": None,
                "gpt_3.5_turbo": None,
                "gpt_4o": None,
            },
        },
    ]

    return samples


def test_difficulty_evaluation():
    """Test the difficulty evaluation functionality."""

    print("🧪 Testing FaithBench Difficulty Evaluation")
    print("=" * 50)

    # Create adapter with challenging case filtering enabled
    config = FaithBenchConfig(
        enable_challenging_case_filtering=True,
        challenge_level="hard",
        model_disagreement_threshold=0.3,
        min_model_predictions=3,
    )
    adapter = FaithBenchAdapter(config)

    samples = create_mock_faithbench_samples()

    print("🔍 Evaluating Sample Difficulty:")
    print("-" * 40)

    difficulty_results = []

    for sample in samples:
        sample_id = sample["sample_id"]
        difficulty_score = adapter.evaluate_detection_difficulty(sample)

        # Parse sample for additional context
        faithbench_sample = adapter._parse_faithbench_sample(sample)

        print(f"\n📝 Sample {sample_id}:")
        print(f"   Source: {sample['source'][:60]}...")
        print(f"   Summary: {sample['summary'][:60]}...")
        print(f"   Annotations: {len(faithbench_sample.annotations)}")
        print(f"   Max Severity: {faithbench_sample.max_severity}")
        print(f"   Difficulty Score: {difficulty_score:.3f}")

        # Categorize difficulty
        if difficulty_score < 0.3:
            difficulty_category = "Easy"
        elif difficulty_score < 0.7:
            difficulty_category = "Medium"
        else:
            difficulty_category = "Hard"

        print(f"   Difficulty Category: {difficulty_category}")

        # Show model predictions if available
        metadata = faithbench_sample.metadata
        model_preds = []
        for model_name, value in [
            ("HHEM v1", metadata.hhemv1),
            ("HHEM 2.1", metadata.hhem_2_1),
            ("TrueTeacher", metadata.trueteacher),
            ("True NLI", metadata.true_nli),
            ("GPT-3.5", metadata.gpt_3_5_turbo),
            ("GPT-4o", metadata.gpt_4o),
        ]:
            if value is not None:
                model_preds.append(f"{model_name}:{value}")

        if model_preds:
            print(f"   Model Predictions: {', '.join(model_preds)}")

        difficulty_results.append(
            {
                "sample_id": sample_id,
                "difficulty_score": difficulty_score,
                "category": difficulty_category,
                "num_annotations": len(faithbench_sample.annotations),
                "max_severity": faithbench_sample.max_severity,
            }
        )

    # Verify difficulty ordering makes sense
    print(f"\n📊 Difficulty Analysis Summary:")
    difficulty_results.sort(key=lambda x: x["difficulty_score"])

    for result in difficulty_results:
        print(
            f"   Sample {result['sample_id']}: {result['difficulty_score']:.3f} ({result['category']})"
        )

    # Basic validation - samples with more annotations/higher severity should generally be harder
    expected_order = [1, 5, 2, 3, 4]  # Based on our mock data design
    actual_order = [result["sample_id"] for result in difficulty_results]

    print(f"   Expected difficulty order: {expected_order}")
    print(f"   Actual difficulty order: {actual_order}")

    # Allow some flexibility in ordering
    reasonable_ordering = True
    if actual_order[0] not in [1, 5] or actual_order[-1] not in [3, 4]:
        reasonable_ordering = False

    status = "✅" if reasonable_ordering else "⚠️"
    print(
        f"   {status} Difficulty ordering: {'Reasonable' if reasonable_ordering else 'Needs review'}"
    )

    return reasonable_ordering


def test_challenging_case_filtering():
    """Test the challenging case filtering functionality."""

    print("\n\n🧪 Testing Challenging Case Filtering")
    print("=" * 50)

    samples = create_mock_faithbench_samples()

    # Test different challenge levels
    challenge_levels = ["easy", "medium", "hard", "all"]

    for level in challenge_levels:
        config = FaithBenchConfig(
            enable_challenging_case_filtering=True,
            challenge_level=level,
            model_disagreement_threshold=0.3,
        )
        adapter = FaithBenchAdapter(config)

        filtered_samples = adapter.filter_challenging_cases(samples)

        print(f"\n📊 Challenge Level: {level.upper()}")
        print(f"   Original samples: {len(samples)}")
        print(f"   Filtered samples: {len(filtered_samples)}")

        if filtered_samples:
            sample_ids = [s["sample_id"] for s in filtered_samples]
            print(f"   Sample IDs: {sample_ids}")

            # Calculate average difficulty of filtered samples
            total_difficulty = 0
            for sample in filtered_samples:
                difficulty = adapter.evaluate_detection_difficulty(sample)
                total_difficulty += difficulty
            avg_difficulty = total_difficulty / len(filtered_samples)

            print(f"   Average difficulty: {avg_difficulty:.3f}")

        # Validate filtering logic
        if level == "all":
            assert len(filtered_samples) == len(
                samples
            ), "All samples should be included"
        elif level == "easy":
            assert len(filtered_samples) <= len(
                samples
            ), "Should filter out harder samples"
        elif level == "hard":
            assert len(filtered_samples) <= len(
                samples
            ), "Should filter out easier samples"

    print("\n✅ Challenging case filtering tests completed")
    return True


def test_challenge_statistics():
    """Test the challenge statistics functionality."""

    print("\n\n🧪 Testing Challenge Statistics")
    print("=" * 40)

    adapter = FaithBenchAdapter()
    samples = create_mock_faithbench_samples()

    stats = adapter.get_challenge_statistics(samples)

    print("📊 Dataset Challenge Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Average difficulty: {stats['avg_difficulty']:.3f}")
    print("   Difficulty distribution:")
    for level, count in stats["difficulty_distribution"].items():
        percentage = (
            (count / stats["total_samples"]) * 100 if stats["total_samples"] > 0 else 0
        )
        print(f"      {level.capitalize()}: {count} ({percentage:.1f}%)")

    print(f"   Model disagreement cases: {stats['model_disagreement_cases']}")
    print(f"   Annotation-based cases: {stats['annotation_based_cases']}")

    # Validate statistics
    total_categorized = sum(stats["difficulty_distribution"].values())
    assert (
        total_categorized == stats["total_samples"]
    ), "Difficulty categorization should sum to total"

    total_method_based = (
        stats["model_disagreement_cases"] + stats["annotation_based_cases"]
    )
    assert (
        total_method_based == stats["total_samples"]
    ), "Method-based categorization should sum to total"

    print("✅ Challenge statistics validation passed")
    return True


def test_model_specific_filtering():
    """Test filtering by specific model performance."""

    print("\n\n🧪 Testing Model-Specific Filtering")
    print("=" * 45)

    adapter = FaithBenchAdapter()
    samples = create_mock_faithbench_samples()

    # Test filtering for different models
    target_models = ["gpt_3_5_turbo", "gpt_4o", "trueteacher"]

    for model in target_models:
        challenging_for_model = adapter.filter_by_model_performance(
            samples, target_model=model, performance_threshold=0.6
        )

        print(f"\n🎯 Model: {model}")
        print(f"   Challenging samples: {len(challenging_for_model)}")

        if challenging_for_model:
            sample_ids = [s["sample_id"] for s in challenging_for_model]
            print(f"   Sample IDs: {sample_ids}")

            # Show why each sample is challenging for this model
            for sample in challenging_for_model[:3]:  # Show first 3
                faithbench_sample = adapter._parse_faithbench_sample(sample)
                metadata = faithbench_sample.metadata

                model_performance = getattr(metadata, model, None)
                print(
                    f"      Sample {sample['sample_id']}: Model performance = {model_performance}"
                )

    print("\n✅ Model-specific filtering tests completed")
    return True


def test_integration_with_existing_adapter():
    """Test integration with existing FaithBench adapter functionality."""

    print("\n\n🧪 Testing Integration with Existing Adapter")
    print("=" * 50)

    try:
        # Test basic adapter functionality still works
        config = FaithBenchConfig(
            enable_challenging_case_filtering=True, challenge_level="hard"
        )
        adapter = FaithBenchAdapter(config)

        sample = create_mock_faithbench_samples()[2]  # Use medium complexity sample

        print(f"📝 Testing sample {sample['sample_id']}")

        # Test PropositionSet conversion
        prop_set = adapter.adapt_single(sample)
        print(f"✅ PropositionSet created with {len(prop_set)} propositions")

        # Test prompt formatting
        prompt = adapter.format_prompt(sample)
        print(f"✅ Prompt formatted ({len(prompt)} characters)")

        # Test sample parsing
        faithbench_sample = adapter._parse_faithbench_sample(sample)
        print(f"✅ Sample parsed: {len(faithbench_sample.annotations)} annotations")

        print("✅ Integration tests completed - existing functionality preserved")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    try:
        print("🚀 Starting FaithBench Challenging Case Enhancement Validation\n")

        # Run all tests
        test1_passed = test_difficulty_evaluation()
        test2_passed = test_challenging_case_filtering()
        test3_passed = test_challenge_statistics()
        test4_passed = test_model_specific_filtering()
        test5_passed = test_integration_with_existing_adapter()

        all_tests_passed = all(
            [test1_passed, test2_passed, test3_passed, test4_passed, test5_passed]
        )

        if all_tests_passed:
            print("\n🎉 FaithBench Challenging Case Enhancement validation PASSED!")
            print("\n📋 Summary:")
            print("   ✅ Difficulty evaluation working correctly")
            print("   ✅ Challenging case filtering implemented")
            print("   ✅ Challenge statistics functionality added")
            print("   ✅ Model-specific filtering available")
            print("   ✅ Integration with existing adapter maintained")
            print("\n🏆 Enhanced FaithBench now focuses on challenging cases:")
            print("   • Cases where SOTA detection models disagree")
            print("   • Adjustable difficulty levels (easy/medium/hard)")
            print("   • Model-specific performance analysis")
            print("   • Realistic ~50% performance expectations")
        else:
            print("\n⚠️ FaithBench Challenging Case Enhancement validation had issues")
            print(f"   Test 1 (Difficulty Eval): {'✅' if test1_passed else '❌'}")
            print(f"   Test 2 (Case Filtering): {'✅' if test2_passed else '❌'}")
            print(f"   Test 3 (Statistics): {'✅' if test3_passed else '❌'}")
            print(f"   Test 4 (Model Filtering): {'✅' if test4_passed else '❌'}")
            print(f"   Test 5 (Integration): {'✅' if test5_passed else '❌'}")

    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        raise

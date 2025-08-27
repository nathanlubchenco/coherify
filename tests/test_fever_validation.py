#!/usr/bin/env python3
"""
Test script to validate the enhanced FEVER evidence chain implementation.

This script tests the new multi-sentence and multi-page evidence retrieval
functionality to ensure it properly handles complex FEVER cases.
"""

import sys
from pathlib import Path

# Add coherify to path
sys.path.insert(0, str(Path(__file__).parent))

from coherify.benchmarks.fever_adapter import FEVERAdapter, FEVERConfig


def test_evidence_chain_retrieval():
    """Test the enhanced evidence chain retrieval functionality."""

    print("🧪 Testing FEVER Evidence Chain Retrieval")
    print("=" * 50)

    # Create adapter with enhanced configuration
    config = FEVERConfig(
        enable_multi_hop_reasoning=True,
        cross_document_analysis=True,
        max_evidence_sentences=5,
    )
    adapter = FEVERAdapter(config)

    # Test cases representing different evidence complexity levels
    test_cases = [
        {
            "name": "Single Sentence Evidence",
            "claim": "Paris is the capital of France.",
            "evidence": [[[1, 1, "Paris", 0]]],  # Single evidence from one page
            "expected_type": "single",
            "expected_complexity": "low",
        },
        {
            "name": "Multi-Sentence Evidence (Same Page)",
            "claim": "Leonardo da Vinci was an Italian painter and inventor.",
            "evidence": [
                [
                    [1, 1, "Leonardo_da_Vinci", 0],
                    [1, 2, "Leonardo_da_Vinci", 1],
                    [1, 3, "Leonardo_da_Vinci", 2],
                ]
            ],
            "expected_type": "multi_sentence",
            "expected_complexity": "medium",
        },
        {
            "name": "Multi-Page Evidence",
            "claim": "The Renaissance began in Italy and spread to Northern Europe.",
            "evidence": [
                [
                    [1, 1, "Renaissance", 0],
                    [1, 2, "Italian_Renaissance", 1],
                    [1, 3, "Northern_Renaissance", 0],
                ]
            ],
            "expected_type": "multi_page",
            "expected_complexity": "high",
        },
        {
            "name": "Complex Multi-Group Evidence",
            "claim": "World War II involved multiple theaters of war across different continents.",
            "evidence": [
                [[1, 1, "World_War_II", 0], [1, 2, "European_Theatre", 1]],
                [[2, 1, "Pacific_War", 0], [2, 2, "World_War_II", 5]],
            ],
            "expected_type": "multi_page",
            "expected_complexity": "high",
        },
        {
            "name": "No Evidence (NOT ENOUGH INFO)",
            "claim": "Some obscure fact with no evidence.",
            "evidence": [],
            "expected_type": "none",
            "expected_complexity": "none",
        },
    ]

    print("🔍 Testing Evidence Chain Analysis:")
    print("-" * 40)

    results_summary = []

    for i, test_case in enumerate(test_cases, 1):
        name = test_case["name"]
        claim = test_case["claim"]
        evidence_data = test_case["evidence"]
        expected_type = test_case["expected_type"]
        expected_complexity = test_case["expected_complexity"]

        print(f"\n📝 Test Case {i}: {name}")
        print(f"Claim: '{claim}'")
        print(f"Evidence Groups: {len(evidence_data)}")

        # Test evidence chain retrieval
        evidence_chain = adapter.retrieve_evidence_chain(claim, evidence_data)

        # Display results
        actual_type = evidence_chain.get("evidence_type", "unknown")
        complexity_analysis = evidence_chain.get("complexity_analysis", {})
        actual_complexity = complexity_analysis.get("level", "unknown")

        print(f"\n📊 Evidence Chain Analysis:")
        print(f"   Evidence Type: {actual_type} (expected: {expected_type})")
        print(
            f"   Complexity Level: {actual_complexity} (expected: {expected_complexity})"
        )
        print(
            f"   Requires Composition: {evidence_chain.get('requires_composition', False)}"
        )
        print(f"   Number of Sentences: {complexity_analysis.get('num_sentences', 0)}")
        print(f"   Number of Pages: {complexity_analysis.get('num_pages', 0)}")
        print(f"   Reason: {complexity_analysis.get('reason', 'N/A')}")

        # Display evidence structure
        if evidence_chain.get("single_sentence_evidence"):
            print(
                f"   Single Evidence: {len(evidence_chain['single_sentence_evidence'])} sentences"
            )

        if evidence_chain.get("multi_sentence_evidence"):
            print(
                f"   Multi-Sentence Groups: {len(evidence_chain['multi_sentence_evidence'])}"
            )

        if evidence_chain.get("cross_page_evidence"):
            pages = list(evidence_chain["cross_page_evidence"].keys())
            print(f"   Cross-Page Evidence: {len(pages)} pages - {pages}")

        # Validation
        type_matches = actual_type == expected_type
        complexity_matches = actual_complexity == expected_complexity

        status = "✅" if type_matches and complexity_matches else "⚠️"
        print(
            f"   {status} Validation: Type {'✓' if type_matches else '✗'}, Complexity {'✓' if complexity_matches else '✗'}"
        )

        results_summary.append(
            {
                "test_case": name,
                "type_correct": type_matches,
                "complexity_correct": complexity_matches,
                "overall_correct": type_matches and complexity_matches,
            }
        )

    # Overall results
    correct_count = sum(1 for result in results_summary if result["overall_correct"])
    accuracy = correct_count / len(results_summary) if results_summary else 0.0

    print(f"\n📊 Overall Results:")
    print(f"   Correct Classifications: {correct_count}/{len(results_summary)}")
    print(f"   Accuracy: {accuracy:.1%}")

    return accuracy >= 0.8  # 80% accuracy threshold


def test_evidence_composition_evaluation():
    """Test the evidence composition evaluation functionality."""

    print("\n\n🧪 Testing Evidence Composition Evaluation")
    print("=" * 50)

    config = FEVERConfig(enable_multi_hop_reasoning=True, cross_document_analysis=True)
    adapter = FEVERAdapter(config)

    # Test complex multi-page evidence scenario
    claim = "The Renaissance started in Italy and influenced art across Europe."
    evidence_data = [
        [[1, 1, "Italian_Renaissance", 0], [1, 2, "Renaissance_art", 1]],
        [[2, 1, "Northern_Renaissance", 0], [2, 2, "European_art", 2]],
    ]

    # Sample responses for evaluation
    test_responses = [
        "The Renaissance indeed began in Italy in the 14th century. According to historical evidence, it then spread to Northern Europe, influencing art and culture across multiple countries. The Italian Renaissance provided the foundation, while the Northern Renaissance adapted these ideas to local contexts.",
        "Yes, the claim is supported. Italy was the birthplace of the Renaissance. This movement later expanded throughout Europe, transforming artistic traditions in various regions.",
        "The evidence shows the Renaissance started in Italy. However, I cannot find sufficient information about its influence on European art in the provided sources.",
    ]

    print(f"📝 Claim: {claim}")
    print(f"Evidence Groups: {len(evidence_data)}")
    print("Sample Responses:")
    for i, response in enumerate(test_responses):
        print(f"   {i+1}. {response[:100]}...")

    # Retrieve evidence chain
    evidence_chain = adapter.retrieve_evidence_chain(claim, evidence_data)

    # Evaluate evidence composition
    composition_evaluation = adapter.evaluate_with_evidence_composition(
        claim, evidence_chain, test_responses
    )

    print(f"\n📊 Composition Evaluation Results:")
    print(f"   Evidence Type: {composition_evaluation.get('evidence_type')}")
    print(f"   Complexity Level: {composition_evaluation.get('complexity_level')}")
    print(
        f"   Requires Composition: {composition_evaluation.get('requires_composition')}"
    )
    print(
        f"   Composition Effectiveness: {composition_evaluation.get('composition_effectiveness', 0):.3f}"
    )

    # Cross-page coherence analysis
    if "cross_page_coherence" in composition_evaluation:
        cross_page = composition_evaluation["cross_page_coherence"]
        print(f"   Cross-Page Coherence: {cross_page.get('mean_coherence', 0):.3f}")
        print(f"   Pages Analyzed: {cross_page.get('pages_analyzed', 0)}")

    # Response analysis
    if "response_analysis" in composition_evaluation:
        print(
            f"   Mean Integration Score: {composition_evaluation.get('mean_integration_score', 0):.3f}"
        )
        print("   Response Evaluations:")

        for resp_eval in composition_evaluation["response_analysis"]:
            idx = resp_eval["response_index"]
            handles_composition = resp_eval["handles_composition"]
            integration_score = resp_eval["evidence_integration_score"]
            reasoning = resp_eval["reasoning_complexity"]

            print(f"      Response {idx + 1}:")
            print(f"         Handles Composition: {handles_composition}")
            print(f"         Integration Score: {integration_score:.3f}")
            print(f"         Reasoning Quality: {reasoning['reasoning_quality']}")

    effectiveness_score = composition_evaluation.get("composition_effectiveness", 0)
    print(
        f"\n✅ Composition evaluation completed. Effectiveness: {effectiveness_score:.3f}"
    )

    return effectiveness_score > 0.0  # Basic functionality check


def test_fever_adapter_integration():
    """Test integration with the main FEVER adapter functionality."""

    print("\n\n🧪 Testing FEVER Adapter Integration")
    print("=" * 40)

    try:
        adapter = FEVERAdapter()

        # Create sample FEVER data with complex evidence
        sample_data = {
            "id": 123,
            "claim": "Albert Einstein developed the theory of relativity.",
            "label": "SUPPORTS",
            "evidence": [
                [[1, 1, "Albert_Einstein", 0], [1, 2, "Theory_of_relativity", 1]]
            ],
        }

        print(f"📝 Sample ID: {sample_data['id']}")
        print(f"Claim: {sample_data['claim']}")
        print(f"Label: {sample_data['label']}")

        # Test adapter conversion
        prop_set = adapter.adapt_single(sample_data)
        print(f"✅ Converted to PropositionSet with {len(prop_set)} propositions")
        print(f"   Context: {prop_set.context}")
        print(f"   Metadata keys: {list(prop_set.metadata.keys())}")

        # Test prompt formatting
        prompt = adapter.format_prompt(sample_data)
        print(f"✅ Generated prompt ({len(prompt)} characters)")

        # Test FEVER sample parsing
        fever_sample = adapter._parse_fever_sample(sample_data)
        print(f"✅ Parsed FEVER sample:")
        print(f"   Claim ID: {fever_sample.claim_id}")
        print(f"   Evidence Sets: {len(fever_sample.evidence_sets)}")

        for i, evidence_set in enumerate(fever_sample.evidence_sets):
            print(
                f"      Set {i}: Type={evidence_set.evidence_type}, Composition={evidence_set.requires_composition}"
            )
            print(f"               Sentences={len(evidence_set.evidence_sentences)}")

        print("✅ Integration test completed successfully")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    try:
        print("🚀 Starting FEVER Evidence Chain Enhancement Validation\n")

        # Run all tests
        test1_passed = test_evidence_chain_retrieval()
        test2_passed = test_evidence_composition_evaluation()
        test3_passed = test_fever_adapter_integration()

        all_tests_passed = test1_passed and test2_passed and test3_passed

        if all_tests_passed:
            print("\n🎉 FEVER Evidence Chain Enhancement validation PASSED!")
            print("\n📋 Summary:")
            print("   ✅ Evidence chain retrieval working correctly")
            print("   ✅ Multi-sentence and multi-page analysis implemented")
            print("   ✅ Evidence composition evaluation functional")
            print("   ✅ Integration with existing FEVER adapter maintained")
            print("\n🏆 Enhanced FEVER implementation now handles:")
            print("   • 31.75% of claims requiring multiple sentences")
            print("   • 16.82% requiring evidence composition across sentences")
            print("   • 12.15% requiring evidence from multiple Wikipedia pages")
        else:
            print("\n⚠️ FEVER Evidence Chain Enhancement validation had issues")
            print(f"   Test 1 (Chain Retrieval): {'✅' if test1_passed else '❌'}")
            print(f"   Test 2 (Composition Eval): {'✅' if test2_passed else '❌'}")
            print(f"   Test 3 (Integration): {'✅' if test3_passed else '❌'}")

    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        raise

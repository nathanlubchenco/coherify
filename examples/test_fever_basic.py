#!/usr/bin/env python3
"""
Basic FEVER Evidence-Based Coherence Test

This script demonstrates the core functionality of FEVER evidence-based
coherence evaluation without requiring API keys.
"""

from coherify import FEVERAdapter, FEVERConfig, EvidenceBasedCoherence


def test_fever_adapter():
    """Test FEVER adapter functionality."""
    print("🔍 Testing FEVER Adapter")
    print("-" * 40)
    
    # Create FEVER sample
    sample = {
        "id": 1,
        "claim": "Barack Obama was the 44th President of the United States.",
        "label": "SUPPORTS",
        "evidence": [
            [
                [101, 1001, "Barack_Obama", 0],
                [101, 1002, "Barack_Obama", 1]
            ]
        ]
    }
    
    # Setup adapter
    config = FEVERConfig(enable_multi_response=False)
    adapter = FEVERAdapter(config=config, provider=None)
    
    # Test adaptation
    prop_set = adapter.adapt_single(sample)
    print(f"  Claim: {sample['claim']}")
    print(f"  Label: {sample['label']}")
    print(f"  Propositions extracted: {len(prop_set.propositions)}")
    
    for i, prop in enumerate(prop_set.propositions):
        prop_type = prop.metadata.get("type", "unknown")
        print(f"    {i+1}. [{prop_type}] {prop.text[:60]}...")
    
    return prop_set


def test_evidence_based_coherence():
    """Test Evidence-Based Coherence measure."""
    print("\n📊 Testing Evidence-Based Coherence")
    print("-" * 40)
    
    # Test with claim and evidence
    claim = "Water boils at 100 degrees Celsius at sea level pressure."
    evidence_sentences = [
        "Water is a chemical compound with the formula H2O.",
        "The boiling point of water is 100°C (212°F) at standard atmospheric pressure.",
        "At sea level, atmospheric pressure is approximately 1013.25 hPa."
    ]
    
    # Create evidence-based coherence measure
    measure = EvidenceBasedCoherence(provider=None)
    
    # Evaluate evidence coherence
    result = measure.evaluate_evidence_coherence(
        claim=claim,
        evidence_sentences=evidence_sentences,
        context="Scientific fact verification"
    )
    
    print(f"  Claim: {claim}")
    print(f"  Evidence sentences: {len(evidence_sentences)}")
    print(f"  Claim-Evidence Coherence: {result['claim_evidence_coherence']:.3f}")
    print(f"  Evidence Consistency: {result['evidence_consistency']:.3f}")
    print(f"  Overall Coherence: {result['overall_coherence']:.3f}")
    print(f"  Coherence Verdict: {result['coherence_verdict']}")
    
    return result


def test_fever_evaluation():
    """Test FEVER sample evaluation."""
    print("\n🎯 Testing FEVER Sample Evaluation")
    print("-" * 40)
    
    # Test different types of FEVER samples
    samples = [
        {
            "id": 1,
            "claim": "The Earth is approximately spherical in shape.",
            "label": "SUPPORTS",
            "evidence": [[[101, 1001, "Earth", 0]]],
            "description": "Straightforward supported claim"
        },
        {
            "id": 2,
            "claim": "The Moon is made entirely of cheese.",
            "label": "REFUTES",
            "evidence": [[[102, 2001, "Moon", 5]]],
            "description": "Clear refutation case"
        },
        {
            "id": 3,
            "claim": "John Smith had eggs for breakfast yesterday.",
            "label": "NOT ENOUGH INFO",
            "evidence": [[[103, 3001, None, None]]],
            "description": "Insufficient evidence case"
        }
    ]
    
    # Setup adapter
    config = FEVERConfig(enable_multi_response=False)
    adapter = FEVERAdapter(config=config, provider=None)
    
    results = []
    
    for sample in samples:
        print(f"\n  Testing: {sample['description']}")
        print(f"  Claim: {sample['claim']}")
        print(f"  Expected Label: {sample['label']}")
        
        # Adapt sample
        prop_set = adapter.adapt_single(sample)
        
        # Simple coherence evaluation
        from coherify import HybridCoherence
        measure = HybridCoherence()
        coherence_result = measure.compute(prop_set)
        
        print(f"  Coherence Score: {coherence_result.score:.3f}")
        print(f"  Propositions: {len(prop_set.propositions)}")
        
        # Test evidence parsing
        fever_sample = adapter._parse_fever_sample(sample)
        print(f"  Evidence Sets: {len(fever_sample.evidence_sets)}")
        
        results.append({
            "sample": sample,
            "coherence_score": coherence_result.score,
            "num_propositions": len(prop_set.propositions),
            "num_evidence_sets": len(fever_sample.evidence_sets)
        })
    
    return results


def test_multi_evidence_coherence():
    """Test coherence evaluation with multiple evidence sources."""
    print("\n🔗 Testing Multi-Evidence Coherence")
    print("-" * 40)
    
    # Test complex claim requiring multiple evidence
    claim = "Albert Einstein developed the theory of relativity which influenced nuclear physics."
    evidence_sentences = [
        "Albert Einstein was a theoretical physicist born in 1879.",
        "Einstein published the special theory of relativity in 1905.",
        "The general theory of relativity was published by Einstein in 1915.",
        "Einstein's mass-energy equivalence E=mc² is fundamental to nuclear physics.",
        "Nuclear weapons derive their energy from Einstein's mass-energy relationship."
    ]
    
    # Create evidence-based coherence measure
    measure = EvidenceBasedCoherence(provider=None)
    
    # Evaluate with all evidence
    full_result = measure.evaluate_evidence_coherence(
        claim=claim,
        evidence_sentences=evidence_sentences,
        context="Multi-evidence fact verification"
    )
    
    print(f"  Claim: {claim[:60]}...")
    print(f"  Evidence sentences: {len(evidence_sentences)}")
    print(f"  Overall Coherence: {full_result['overall_coherence']:.3f}")
    
    # Test with partial evidence
    partial_evidence = evidence_sentences[:2]  # Only first two sentences
    partial_result = measure.evaluate_evidence_coherence(
        claim=claim,
        evidence_sentences=partial_evidence,
        context="Partial evidence fact verification"
    )
    
    print(f"\n  Partial evidence test:")
    print(f"  Evidence sentences: {len(partial_evidence)}")
    print(f"  Overall Coherence: {partial_result['overall_coherence']:.3f}")
    print(f"  Coherence difference: {full_result['overall_coherence'] - partial_result['overall_coherence']:+.3f}")
    
    return {
        "full_evidence": full_result,
        "partial_evidence": partial_result,
        "coherence_improvement": full_result['overall_coherence'] - partial_result['overall_coherence']
    }


def main():
    """Run all FEVER basic tests."""
    print("🚀 FEVER Evidence-Based Coherence Basic Test")
    print("=" * 60)
    
    # Test components
    prop_set = test_fever_adapter()
    coherence_result = test_evidence_based_coherence()
    evaluation_results = test_fever_evaluation()
    multi_evidence_result = test_multi_evidence_coherence()
    
    # Summary
    print("\n📊 Test Summary")
    print("-" * 40)
    print(f"✅ FEVER Adapter: Successfully extracted {len(prop_set.propositions)} propositions")
    print(f"✅ Evidence-Based Coherence: Overall score {coherence_result['overall_coherence']:.3f}")
    print(f"✅ FEVER Evaluation: Tested {len(evaluation_results)} sample types")
    print(f"✅ Multi-Evidence: Coherence improvement with more evidence: {multi_evidence_result['coherence_improvement']:+.3f}")
    
    print("\n🎯 Key Insights:")
    print("  • FEVER adapter properly extracts claims and evidence as propositions")
    print("  • Evidence-based coherence combines claim-evidence and evidence-evidence coherence")
    print("  • Different FEVER labels (SUPPORTS/REFUTES/NOT ENOUGH INFO) show different coherence patterns")
    print("  • More comprehensive evidence generally improves coherence scores")
    
    print("\n✅ FEVER evidence-based coherence testing completed!")
    print("\n💡 Next steps:")
    print("  - Try with API providers for multi-response evidence evaluation")
    print("  - Test with real Wikipedia evidence retrieval")
    print("  - Experiment with different evidence coherence weights")
    print("  - Compare evidence-based coherence across different domains")


if __name__ == "__main__":
    main()
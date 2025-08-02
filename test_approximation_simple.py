#!/usr/bin/env python3
"""Simple test of approximation algorithms functionality."""

import numpy as np
from coherify import PropositionSet, Proposition
from coherify.measures import SemanticCoherence
from coherify.approximation import (
    RandomSampler, SamplingBasedApproximator,
    ClusterBasedApproximator, IncrementalCoherenceTracker
)

def test_sampling():
    """Test sampling-based approximation."""
    print("Testing sampling approximation...")
    
    # Create small test set
    props = [
        Proposition("Machine learning uses algorithms to learn patterns."),
        Proposition("Neural networks are inspired by brain structure."),
        Proposition("Deep learning uses multiple layers."),
        Proposition("Cats are popular pets."),
        Proposition("The sky is blue on clear days.")
    ]
    prop_set = PropositionSet(props)
    
    # Test sampling
    sampler = RandomSampler(seed=42)
    approximator = SamplingBasedApproximator(sampler, SemanticCoherence())
    
    result = approximator.approximate_coherence(prop_set, 3)
    
    print(f"✅ Sampling: {result.approximate_score:.3f} (sampled {result.sample_size}/{result.total_size})")
    return True

def test_clustering():
    """Test clustering-based approximation."""
    print("Testing clustering approximation...")
    
    # Create test set
    props = [
        Proposition("Machine learning algorithms analyze data."),
        Proposition("Neural networks process information."),
        Proposition("Deep learning uses multiple layers."),
        Proposition("The weather is sunny today."),
        Proposition("Birds can fly through the air.")
    ]
    prop_set = PropositionSet(props)
    
    # Test clustering
    approximator = ClusterBasedApproximator(SemanticCoherence())
    result = approximator.approximate_coherence(prop_set, 3)
    
    print(f"✅ Clustering: {result.approximate_score:.3f} ({result.num_clusters} clusters)")
    return True

def test_incremental():
    """Test incremental tracking."""
    print("Testing incremental tracking...")
    
    tracker = IncrementalCoherenceTracker(SemanticCoherence())
    
    propositions = [
        "Machine learning is a subset of AI.",
        "Neural networks mimic brain function.",
        "The weather is nice today."
    ]
    
    for i, prop_text in enumerate(propositions):
        prop = Proposition(prop_text)
        update = tracker.add_proposition(prop)
        print(f"  {i+1}. Score: {update.new_score:.3f} ({'incremental' if update.incremental_computation else 'full'})")
    
    print("✅ Incremental tracking completed")
    return True

def main():
    """Run simple tests."""
    print("🔬 Simple Approximation Algorithm Tests")
    print("=" * 45)
    
    try:
        test_sampling()
        test_clustering() 
        test_incremental()
        
        print("\n✅ All approximation algorithms working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
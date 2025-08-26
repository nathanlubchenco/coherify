#!/usr/bin/env python3
"""
Test the 3-stage pipeline with detailed progress monitoring.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from coherify.generation.model_runner import ModelRunner, KPassGenerator
from coherify.evaluators.response_selectors import (
    MajorityVotingSelector, 
    CoherenceSelector,
    StageComparator
)
from coherify.measures import SemanticCoherence
from coherify.benchmarks.official.truthfulqa_official import TruthfulQAOfficialEvaluator
from datasets import load_dataset


def test_pipeline_with_progress(samples=3, k_responses=3):
    """Test the full pipeline with detailed progress output."""
    
    print("🚀 Testing 3-Stage Pipeline with Progress Monitoring")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OPENAI_API_KEY found!")
        return
    print("✅ API key found")
    
    # Load data
    print(f"\n📚 Loading {samples} TruthfulQA samples...")
    dataset = load_dataset("truthful_qa", "generation")
    data = list(dataset["validation"].select(range(samples)))
    
    # Setup model
    model_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 100
    }
    runner = ModelRunner(model_config)
    
    # Stage 1: Single response
    print(f"\n📝 STAGE 1: Generating single response per question")
    print("-" * 40)
    stage1_predictions = []
    for i, sample in enumerate(data):
        print(f"  Q{i+1}/{samples}: Generating...", end="", flush=True)
        result = runner.generate_response(sample["question"])
        stage1_predictions.append(result.text)
        print(f" ✓ ({len(result.text)} chars)")
    
    # Stage 2: K-pass with majority voting
    print(f"\n📝 STAGE 2: Generating K={k_responses} responses with majority voting")
    print("-" * 40)
    k_generator = KPassGenerator(runner, k=k_responses)
    stage2_predictions = []
    
    for i, sample in enumerate(data):
        print(f"  Q{i+1}/{samples}: Generating {k_responses} responses...", end="", flush=True)
        k_results = runner.generate_k_responses(sample["question"], k=k_responses)
        
        print(" Selecting by majority...", end="", flush=True)
        selector = MajorityVotingSelector()
        selection = selector.select(k_results)
        stage2_predictions.append(selection.selected_response)
        print(f" ✓")
    
    # Stage 3: K-pass with coherence selection
    print(f"\n📝 STAGE 3: Generating K={k_responses} responses with coherence selection")
    print("-" * 40)
    stage3_predictions = []
    
    for i, sample in enumerate(data):
        print(f"  Q{i+1}/{samples}: Generating {k_responses} responses...", end="", flush=True)
        k_results = runner.generate_k_responses(sample["question"], k=k_responses)
        
        print(" Computing coherence...", end="", flush=True)
        selector = CoherenceSelector(
            coherence_measure=SemanticCoherence(),
            question=sample["question"]
        )
        selection = selector.select(k_results)
        stage3_predictions.append(selection.selected_response)
        score = getattr(selection, 'score', selection.metadata.get('coherence_score', 0.0))
        print(f" ✓ (score: {score:.2f})")
    
    # Compare stages
    print(f"\n📊 EVALUATING & COMPARING ALL STAGES")
    print("=" * 60)
    print("⚠️  Each stage evaluation requires GPT-4 API calls")
    print("    This may take ~10 seconds per stage\n")
    
    evaluator = TruthfulQAOfficialEvaluator(method="auto")
    comparator = StageComparator(evaluator)
    
    # Show what method is being used
    print(f"📋 Evaluation method: {evaluator.actual_method}")
    
    # Time the comparison
    start_time = time.time()
    comparison = comparator.compare_stages(
        samples=data,
        stage1_predictions=stage1_predictions,
        stage2_predictions=stage2_predictions,
        stage3_predictions=stage3_predictions,
        verbose=True
    )
    elapsed = time.time() - start_time
    
    print(f"\n✅ Comparison completed in {elapsed:.1f}s")
    
    # Show final results
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Stage 1 (Baseline):        {comparison['stage1']['score']:.1%}")
    print(f"Stage 2 (Majority Voting): {comparison['stage2']['score']:.1%} (+{comparison['stage2']['improvement']:.1%})")
    print(f"Stage 3 (Coherence):       {comparison['stage3']['score']:.1%} (+{comparison['stage3']['improvement']:.1%})")
    print(f"\n🎯 Coherence advantage over majority: {comparison['coherence_advantage']:.1%}")
    
    if comparison['coherence_advantage'] > 0:
        print("✅ Coherence selection outperformed majority voting!")
    elif comparison['coherence_advantage'] < 0:
        print("⚠️  Majority voting outperformed coherence (may need tuning)")
    else:
        print("➖ Coherence and majority voting performed equally")
    
    return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3, help="Number of samples")
    parser.add_argument("--k-responses", type=int, default=3, help="K responses per question")
    
    args = parser.parse_args()
    
    test_pipeline_with_progress(args.samples, args.k_responses)
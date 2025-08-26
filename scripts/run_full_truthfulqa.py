#!/usr/bin/env python3
"""
Run full TruthfulQA dataset without sampling.

This script runs the complete TruthfulQA validation set through the 3-stage pipeline.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset


def main():
    """Run full TruthfulQA evaluation."""
    
    parser = argparse.ArgumentParser(description="Run full TruthfulQA dataset")
    parser.add_argument("--model", type=str, default="gpt4-mini", help="Model to use")
    parser.add_argument("--k-responses", type=int, default=5, help="K responses for stages 2 and 3")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (None for full dataset)")
    parser.add_argument("--batch-size", type=int, default=10, help="Process in batches to show progress")
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OPENAI_API_KEY found!")
        print("   Run: export OPENAI_API_KEY=your-key-here")
        return 1
    
    print("🚀 Full TruthfulQA Evaluation")
    print("=" * 60)
    
    # Load full dataset
    print("📚 Loading TruthfulQA validation set...")
    dataset = load_dataset("truthful_qa", "generation")
    validation_data = dataset["validation"]
    
    total_samples = len(validation_data)
    if args.limit:
        total_samples = min(args.limit, total_samples)
        validation_data = validation_data.select(range(total_samples))
    
    print(f"✅ Loaded {total_samples} samples")
    print(f"   Categories: {len(set(validation_data['category']))}")
    
    # Show category distribution
    from collections import Counter
    categories = Counter(validation_data['category'])
    print("\n📊 Category Distribution:")
    for cat, count in categories.most_common(5):
        print(f"   {cat}: {count} ({count/total_samples*100:.1f}%)")
    if len(categories) > 5:
        print(f"   ... and {len(categories)-5} more categories")
    
    print(f"\n⚙️  Configuration:")
    print(f"   Model: {args.model}")
    print(f"   K-responses: {args.k_responses}")
    print(f"   Batch size: {args.batch_size}")
    
    # Process in batches
    print(f"\n📝 Processing {total_samples} samples in batches of {args.batch_size}...")
    print("   This will take approximately {:.0f} minutes".format(
        total_samples * 15 / 60  # ~15 seconds per sample with K=5
    ))
    
    # Import from examples directory
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    from run_full_pipeline_comparison import run_full_pipeline_comparison
    from coherify.utils.model_configs import load_model_config
    
    # Load model config
    model_config = load_model_config(args.model)
    
    all_results = []
    for i in range(0, total_samples, args.batch_size):
        batch_end = min(i + args.batch_size, total_samples)
        batch_samples = validation_data.select(range(i, batch_end))
        
        print(f"\n🔄 Batch {i//args.batch_size + 1}/{(total_samples + args.batch_size - 1)//args.batch_size}")
        print(f"   Samples {i+1}-{batch_end} of {total_samples}")
        
        try:
            result = run_full_pipeline_comparison(
                model_config=model_config,
                n_samples=len(batch_samples),
                k_responses=args.k_responses,
                samples=batch_samples,
                verbose=False
            )
            
            all_results.append(result)
            
            # Show running average
            avg_stage1 = sum(r['stage1']['score'] for r in all_results) / len(all_results)
            avg_stage2 = sum(r['stage2']['score'] for r in all_results) / len(all_results)
            avg_stage3 = sum(r['stage3']['score'] for r in all_results) / len(all_results)
            
            print(f"   📊 Running Averages:")
            print(f"      Stage 1: {avg_stage1:.1%}")
            print(f"      Stage 2: {avg_stage2:.1%} (+{avg_stage2-avg_stage1:.1%})")
            print(f"      Stage 3: {avg_stage3:.1%} (+{avg_stage3-avg_stage1:.1%})")
            
        except Exception as e:
            print(f"   ❌ Batch failed: {e}")
            continue
    
    # Final summary
    if all_results:
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)
        
        total_processed = sum(r.get('n_samples', 0) for r in all_results)
        avg_stage1 = sum(r['stage1']['score'] for r in all_results) / len(all_results)
        avg_stage2 = sum(r['stage2']['score'] for r in all_results) / len(all_results)
        avg_stage3 = sum(r['stage3']['score'] for r in all_results) / len(all_results)
        
        print(f"Samples Processed: {total_processed}/{total_samples}")
        print(f"\nPerformance:")
        print(f"  Stage 1 (Baseline):        {avg_stage1:.1%}")
        print(f"  Stage 2 (Majority Voting): {avg_stage2:.1%} (+{avg_stage2-avg_stage1:.1%})")
        print(f"  Stage 3 (Coherence):       {avg_stage3:.1%} (+{avg_stage3-avg_stage1:.1%})")
        print(f"\n🎯 Coherence Advantage: {avg_stage3-avg_stage2:.1%}")
        
        # Compare to research expectations
        print(f"\n📚 Research Context:")
        print(f"  Human Performance: 94%")
        print(f"  GPT-3 (best published): 58%")
        print(f"  Our Result: {avg_stage3:.1%}")
        
        if avg_stage3 > avg_stage2:
            print("\n✅ SUCCESS: Coherence selection improved performance!")
        elif avg_stage3 == avg_stage2:
            print("\n➖ NEUTRAL: Coherence matched majority voting")
        else:
            print("\n⚠️  NEEDS TUNING: Majority voting outperformed coherence")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
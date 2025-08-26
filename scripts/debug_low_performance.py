#!/usr/bin/env python3
"""
Debug script to investigate low performance scores.

This will run a small test with verbose output to see what's happening.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from coherify.generation.model_runner import ModelRunner
from coherify.benchmarks.official.truthfulqa_official import TruthfulQAOfficialEvaluator
from datasets import load_dataset


def debug_single_question():
    """Debug a single question to see what's happening."""
    
    print("=" * 60)
    print("🔍 Debugging Single Question")
    print("=" * 60)
    
    # Load one question
    dataset = load_dataset("truthful_qa", "generation")
    sample = dataset["validation"][0]
    
    print(f"\n📝 Question: {sample['question']}")
    print(f"✅ Best answer: {sample['best_answer']}")
    print(f"✅ Correct answers: {sample['correct_answers'][:3]}...")
    print(f"❌ Incorrect answers: {sample['incorrect_answers'][:3]}...")
    
    # Generate response
    model_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    runner = ModelRunner(model_config)
    
    print(f"\n🤖 Generating response...")
    result = runner.generate_response(sample['question'])
    print(f"📤 Model response: '{result.text}'")
    
    # Evaluate
    evaluator = TruthfulQAOfficialEvaluator(method="auto")
    
    print(f"\n📊 Evaluating...")
    eval_result = evaluator.evaluate_dataset([result.text], [sample])
    
    print(f"✅ Truthful: {eval_result.truthful_score:.1%}")
    print(f"ℹ️  Informative: {eval_result.informative_score:.1%}")
    print(f"📋 Method: {eval_result.method}")
    
    if eval_result.warning:
        print(f"⚠️  {eval_result.warning}")
        
    # Show detailed evaluation
    if eval_result.per_sample_results:
        detail = eval_result.per_sample_results[0]
        print(f"\n📝 Detailed result:")
        print(f"  Is truthful: {detail.get('is_truthful', 'N/A')}")
        print(f"  Is informative: {detail.get('is_informative', 'N/A')}")
        if 'confidence' in detail:
            print(f"  Confidence: {detail['confidence']:.2f}")
            
    return eval_result.truthful_score > 0


def test_multiple_questions(n=5):
    """Test multiple questions to see the pattern."""
    
    print("\n" + "=" * 60)
    print(f"🔍 Testing {n} Questions")
    print("=" * 60)
    
    # Load questions
    dataset = load_dataset("truthful_qa", "generation")
    samples = list(dataset["validation"].select(range(n)))
    
    # Setup model
    model_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    runner = ModelRunner(model_config)
    evaluator = TruthfulQAOfficialEvaluator(method="auto")
    
    predictions = []
    for i, sample in enumerate(samples):
        print(f"\n📝 Q{i+1}: {sample['question'][:100]}...")
        result = runner.generate_response(sample['question'])
        print(f"   A: {result.text[:100]}...")
        predictions.append(result.text)
        
    # Evaluate all
    print(f"\n📊 Evaluating all {n} responses...")
    eval_result = evaluator.evaluate_dataset(predictions, samples)
    
    print(f"\n📈 Results:")
    print(f"  Truthful: {eval_result.truthful_score:.1%}")
    print(f"  Informative: {eval_result.informative_score:.1%}")
    print(f"  Method: {eval_result.method}")
    
    # Show which ones were marked truthful
    truthful_count = 0
    for i, result in enumerate(eval_result.per_sample_results):
        if result.get('is_truthful'):
            truthful_count += 1
            print(f"  ✅ Q{i+1} marked truthful")
        else:
            print(f"  ❌ Q{i+1} marked false")
            
    print(f"\n📊 Summary: {truthful_count}/{n} truthful ({truthful_count/n*100:.1f}%)")
    
    return eval_result


def check_evaluation_method():
    """Check what evaluation method is being used."""
    
    print("\n" + "=" * 60)
    print("🔍 Checking Evaluation Method")
    print("=" * 60)
    
    evaluator = TruthfulQAOfficialEvaluator(method="auto")
    print(f"  Method selected: {evaluator.actual_method}")
    
    if evaluator.actual_method == "gpt4-judge":
        print("  ✅ Using GPT-4 as judge (good!)")
    elif evaluator.actual_method == "embedding-similarity":
        print("  ⚠️  Using embedding similarity (less accurate)")
        print("     Set OPENAI_API_KEY for better evaluation")
    else:
        print(f"  ℹ️  Using {evaluator.actual_method}")
        
    # Test the GPT-4 judge directly
    if evaluator.actual_method == "gpt4-judge":
        from coherify.benchmarks.official.truthfulqa_gpt4_judge import TruthfulQAGPT4Judge
        
        judge = TruthfulQAGPT4Judge(model="gpt-4o-mini")
        
        # Test with a simple example
        test_sample = {
            "question": "What is 2+2?",
            "best_answer": "4",
            "correct_answers": ["4", "four"],
            "incorrect_answers": ["5", "22"]
        }
        
        result = judge.evaluate_single("4", test_sample)
        print(f"\n  Test evaluation of '2+2=4':")
        print(f"    Truthful: {result.is_truthful}")
        print(f"    Confidence: {result.truthfulness_confidence:.2f}")


def main():
    """Run all debug tests."""
    
    print("🔍 TruthfulQA Low Performance Debugger")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OPENAI_API_KEY found!")
        print("   Run: source ~/.zshrc")
        return
    else:
        print("✅ API key found")
        
    # Check evaluation method
    check_evaluation_method()
    
    # Debug single question
    print("\n" + "=" * 60)
    success = debug_single_question()
    
    if not success:
        print("\n⚠️  Single question failed. Checking pattern...")
        
    # Test multiple
    result = test_multiple_questions(5)
    
    # Diagnosis
    print("\n" + "=" * 60)
    print("📋 Diagnosis")
    print("=" * 60)
    
    if result.truthful_score < 0.2:
        print("❌ Very low performance detected. Possible causes:")
        print("  1. GPT-4 judge prompts may be too strict")
        print("  2. Model responses may be too short/uninformative")
        print("  3. Temperature may be causing inconsistent responses")
        print("\nTry:")
        print("  - Lower temperature (0.3 instead of 0.7)")
        print("  - Longer max_tokens (200 instead of 100)")
        print("  - Check if responses actually answer the questions")
    elif result.truthful_score < 0.4:
        print("⚠️  Below expected performance. This might be normal for:")
        print("  - Particularly tricky questions")
        print("  - Categories like 'Misconceptions' or 'Conspiracies'")
        print("  - Small sample sizes")
    else:
        print("✅ Performance looks reasonable!")
        
    print("\nNext step: Run with more samples to get stable results")
    print("  make benchmark-full-pipeline MODEL=gpt4-mini SAMPLES=50 K_RUNS=3")


if __name__ == "__main__":
    main()
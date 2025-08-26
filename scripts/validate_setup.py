#!/usr/bin/env python3
"""
Setup validation script to diagnose common issues.

Run this to check if your environment is properly configured.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 Checking API Keys...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if openai_key:
        print(f"  ✅ OpenAI API key found (starts with {openai_key[:7]}...)")
    else:
        print("  ❌ OpenAI API key NOT found (set OPENAI_API_KEY)")
        
    if anthropic_key:
        print(f"  ✅ Anthropic API key found (starts with {anthropic_key[:7]}...)")
    else:
        print("  ⚠️  Anthropic API key not found (optional)")
        
    return bool(openai_key or anthropic_key)


def test_generation():
    """Test if model generation works."""
    print("\n🤖 Testing Model Generation...")
    
    try:
        from coherify.generation.model_runner import ModelRunner
        
        # Try with mock first
        print("  Testing mock model...")
        mock_config = {"provider": "mock", "model": "test"}
        mock_runner = ModelRunner(mock_config)
        mock_response = mock_runner.generate_response("Test prompt")
        print(f"    ✅ Mock generation works: '{mock_response.text[:50]}...'")
        
        # Try with real model if API key available
        if os.getenv("OPENAI_API_KEY"):
            print("  Testing OpenAI model...")
            openai_config = {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            try:
                openai_runner = ModelRunner(openai_config)
                response = openai_runner.generate_response("What is 2+2?")
                
                if response.text and len(response.text) > 0:
                    print(f"    ✅ OpenAI generation works: '{response.text[:50]}...'")
                    return True
                else:
                    print(f"    ❌ OpenAI returned empty response")
                    return False
                    
            except Exception as e:
                print(f"    ❌ OpenAI generation failed: {e}")
                return False
        else:
            print("  ⚠️  Skipping OpenAI test (no API key)")
            
    except Exception as e:
        print(f"  ❌ Generation test failed: {e}")
        return False
        
    return True


def test_evaluation():
    """Test if evaluation works."""
    print("\n📊 Testing Evaluation...")
    
    try:
        from coherify.benchmarks.official.truthfulqa_official import TruthfulQAOfficialEvaluator
        
        # Create test data
        predictions = ["The sky is blue because of Rayleigh scattering"]
        samples = [{
            "question": "Why is the sky blue?",
            "best_answer": "The sky is blue because of Rayleigh scattering",
            "correct_answers": ["Rayleigh scattering", "Light scattering"],
            "incorrect_answers": ["Because of the ocean", "Because of reflections"],
            "category": "Science"
        }]
        
        # Test evaluator
        evaluator = TruthfulQAOfficialEvaluator(method="auto")
        result = evaluator.evaluate_dataset(predictions, samples)
        
        print(f"  Evaluation method: {result.method}")
        print(f"  Truthfulness score: {result.truthful_score:.1%}")
        
        if result.truthful_score > 0:
            print("  ✅ Evaluation works correctly")
            return True
        else:
            print("  ⚠️  Evaluation returned 0% (may need real API)")
            return False
            
    except Exception as e:
        print(f"  ❌ Evaluation test failed: {e}")
        return False


def test_coherence():
    """Test if coherence measures work."""
    print("\n🧠 Testing Coherence Measures...")
    
    try:
        from coherify.measures import SemanticCoherence
        from coherify.core.base import Proposition, PropositionSet
        
        # Create test proposition set
        props = PropositionSet(
            propositions=[
                Proposition(text="The sky is blue"),
                Proposition(text="The ocean is blue"),
                Proposition(text="Blue is a color")
            ]
        )
        
        # Test semantic coherence
        measure = SemanticCoherence()
        result = measure.compute(props)
        
        print(f"  Coherence score: {result.score:.3f}")
        
        if 0 < result.score < 1:
            print("  ✅ Coherence calculation works")
            return True
        else:
            print("  ❌ Unexpected coherence score")
            return False
            
    except Exception as e:
        print(f"  ❌ Coherence test failed: {e}")
        return False


def test_pipeline():
    """Test if full pipeline can run."""
    print("\n🚀 Testing Full Pipeline...")
    
    try:
        from coherify.evaluators.response_selectors import (
            MajorityVotingSelector,
            CoherenceSelector
        )
        from coherify.generation.model_runner import GenerationResult
        
        # Create fake responses
        responses = [
            GenerationResult("Answer A", "test", "mock", 0.7, 0.1),
            GenerationResult("Answer B", "test", "mock", 0.7, 0.1),
            GenerationResult("Answer A", "test", "mock", 0.7, 0.1),
        ]
        
        # Test majority voting
        majority = MajorityVotingSelector()
        result = majority.select(responses)
        print(f"  Majority selected: '{result.selected_response}' (confidence: {result.confidence:.2f})")
        
        # Test coherence selection
        coherence = CoherenceSelector()
        result = coherence.select(responses)
        print(f"  Coherence selected: '{result.selected_response}' (method: {result.method})")
        
        print("  ✅ Pipeline components work")
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("🔍 Coherify Setup Validation")
    print("=" * 60)
    
    results = {
        "API Keys": check_api_keys(),
        "Generation": test_generation(),
        "Evaluation": test_evaluation(),
        "Coherence": test_coherence(),
        "Pipeline": test_pipeline()
    }
    
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    
    for component, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {component}")
        
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All components working! You're ready to run benchmarks.")
        print("\nNext step:")
        print("  make benchmark-full-pipeline MODEL=gpt4-mini SAMPLES=10 K_RUNS=3")
    else:
        print("\n⚠️  Some components need attention.")
        print("\nRecommended fixes:")
        if not results["API Keys"]:
            print("  1. Set OPENAI_API_KEY environment variable")
        if not results["Generation"]:
            print("  2. Check API key is valid and has credits")
        if not results["Evaluation"]:
            print("  3. Install evaluation dependencies: pip install bleurt-pytorch")
            
    print("\nFor more help, see: docs/TROUBLESHOOTING.md")


if __name__ == "__main__":
    main()
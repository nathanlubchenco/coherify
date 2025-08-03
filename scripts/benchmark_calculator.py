#!/usr/bin/env python3
"""
Benchmark Calculator for Coherify TruthfulQA Operations

This script helps you estimate time, cost, and resource requirements
for running TruthfulQA benchmarks with different configurations.
"""

import argparse
import time
from typing import Dict, Any

# API Pricing (per 1K tokens, as of 2024)
API_PRICING = {
    'openai': {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'o3-mini': {'input': 0.015, 'output': 0.06},  # Reasoning models
        'o3': {'input': 0.06, 'output': 0.24}
    },
    'anthropic': {
        'claude-3.5-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125}
    }
}

# Dataset characteristics
TRUTHFULQA_VALIDATION_SIZE = 817
AVG_TOKENS_PER_SAMPLE = 26
DOWNLOAD_TIME_SECONDS = 2
DATASET_SIZE_MB = 0.5

# Performance characteristics (measured)
LOCAL_PERFORMANCE = {
    'semantic': {'time_per_sample': 0.076, 'memory_gb': 4},
    'hybrid': {'time_per_sample': 0.443, 'memory_gb': 8}
}


def calculate_local_processing(sample_size: int, measure_type: str = 'hybrid') -> Dict[str, Any]:
    """Calculate local processing estimates."""
    perf = LOCAL_PERFORMANCE[measure_type]
    
    total_time = sample_size * perf['time_per_sample']
    
    return {
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'time_per_sample': perf['time_per_sample'],
        'memory_required_gb': perf['memory_gb'],
        'throughput_samples_per_minute': 60 / perf['time_per_sample'],
        'cost': 0.0  # Local processing is free
    }


def calculate_api_processing(
    sample_size: int,
    provider: str,
    model: str,
    generations_per_sample: int = 1,
    temperature_variants: int = 1,
    answer_expansion: bool = False,
    reasoning_enabled: bool = False
) -> Dict[str, Any]:
    """Calculate API processing estimates."""
    if provider not in API_PRICING or model not in API_PRICING[provider]:
        raise ValueError(f"Unknown provider/model: {provider}/{model}")
    
    pricing = API_PRICING[provider][model]
    
    # Calculate API calls
    api_calls_per_sample = generations_per_sample * temperature_variants
    if answer_expansion:
        api_calls_per_sample += 1
    
    total_api_calls = sample_size * api_calls_per_sample
    
    # Calculate tokens
    input_tokens_per_call = AVG_TOKENS_PER_SAMPLE
    output_tokens_per_call = 14  # Average answer length
    
    if reasoning_enabled:
        output_tokens_per_call *= 3  # Reasoning models generate more
    
    if answer_expansion:
        output_tokens_per_call *= 1.5  # Expanded answers are longer
    
    total_input_tokens = total_api_calls * input_tokens_per_call
    total_output_tokens = total_api_calls * output_tokens_per_call
    
    # Calculate costs
    input_cost = (total_input_tokens / 1000) * pricing['input']
    output_cost = (total_output_tokens / 1000) * pricing['output']
    total_cost = input_cost + output_cost
    
    # Estimate time (includes API latency)
    avg_api_latency = 2.0  # seconds per call
    total_time = total_api_calls * avg_api_latency
    
    return {
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'api_calls': total_api_calls,
        'api_calls_per_sample': api_calls_per_sample,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'cost_per_sample': total_cost / sample_size if sample_size > 0 else 0
    }


def print_comparison_table(sample_size: int):
    """Print a comparison table of different approaches."""
    print(f"\n📊 Comparison for {sample_size} samples:")
    print("=" * 80)
    
    # Local processing options
    local_semantic = calculate_local_processing(sample_size, 'semantic')
    local_hybrid = calculate_local_processing(sample_size, 'hybrid')
    
    # API processing options
    api_configs = [
        ('OpenAI', 'gpt-3.5-turbo', 1, 1, False, False, 'Budget API'),
        ('Anthropic', 'claude-3-haiku', 1, 1, False, False, 'Budget API'),
        ('OpenAI', 'gpt-4-turbo', 1, 2, True, False, 'Standard API'),
        ('Anthropic', 'claude-3.5-sonnet', 2, 2, True, False, 'Enhanced API'),
        ('OpenAI', 'o3-mini', 2, 3, True, True, 'Reasoning API')
    ]
    
    results = []
    
    # Add local results
    results.append(('Local Semantic', local_semantic['total_time_minutes'], 0.0, 'CPU'))
    results.append(('Local Hybrid', local_hybrid['total_time_minutes'], 0.0, 'CPU+RAM'))
    
    # Add API results
    for provider, model, gens, temps, expansion, reasoning, desc in api_configs:
        try:
            api_result = calculate_api_processing(
                sample_size, provider, model, gens, temps, expansion, reasoning
            )
            results.append((
                f"{desc}",
                api_result['total_time_minutes'],
                api_result['total_cost'],
                f"{api_result['api_calls']} calls"
            ))
        except ValueError:
            continue
    
    # Print table
    print(f"{'Method':<20} {'Time (min)':<12} {'Cost ($)':<10} {'Notes':<15}")
    print("-" * 80)
    
    for method, time_min, cost, notes in results:
        print(f"{method:<20} {time_min:>8.1f}    {cost:>8.2f}    {notes:<15}")


def print_scaling_analysis():
    """Print scaling analysis for different dataset sizes."""
    print("\n📈 Scaling Analysis:")
    print("=" * 60)
    
    sizes = [10, 50, 100, 817, 2000, 5000]
    
    print(f"{'Samples':<10} {'Local (min)':<12} {'Budget API ($)':<15} {'Premium API ($)':<15}")
    print("-" * 60)
    
    for size in sizes:
        local = calculate_local_processing(size, 'hybrid')
        
        try:
            budget_api = calculate_api_processing(size, 'anthropic', 'claude-3-haiku', 1, 1, False, False)
            premium_api = calculate_api_processing(size, 'openai', 'gpt-4-turbo', 2, 2, True, False)
            
            print(f"{size:<10} {local['total_time_minutes']:>8.1f}    "
                  f"{budget_api['total_cost']:>11.2f}     "
                  f"{premium_api['total_cost']:>11.2f}")
        except ValueError:
            print(f"{size:<10} {local['total_time_minutes']:>8.1f}    {'N/A':<15} {'N/A':<15}")


def print_recommendations(sample_size: int):
    """Print recommendations based on use case."""
    print(f"\n💡 Recommendations for {sample_size} samples:")
    print("=" * 50)
    
    # Calculate costs for common scenarios
    local_time = calculate_local_processing(sample_size, 'hybrid')['total_time_minutes']
    
    print(f"🎯 **Quick Testing** (Development)")
    print(f"   Method: Local HybridCoherence")
    print(f"   Time: {local_time:.1f} minutes")
    print(f"   Cost: Free")
    print(f"   Command: python examples/run_truthfulqa_benchmark.py --sample-size {sample_size}")
    
    if sample_size <= 100:
        try:
            budget_api = calculate_api_processing(sample_size, 'anthropic', 'claude-3-haiku')
            print(f"\n💰 **Budget API Testing**")
            print(f"   Method: Claude-3 Haiku")
            print(f"   Time: {budget_api['total_time_minutes']:.1f} minutes")
            print(f"   Cost: ${budget_api['total_cost']:.3f}")
            print(f"   Command: python examples/run_truthfulqa_benchmark.py --use-api --sample-size {sample_size}")
        except ValueError:
            pass
    
    if sample_size >= 50:
        try:
            premium_api = calculate_api_processing(sample_size, 'openai', 'gpt-4-turbo', 2, 2, True, False)
            print(f"\n🏆 **Comprehensive Evaluation**")
            print(f"   Method: GPT-4 Turbo Enhanced")
            print(f"   Time: {premium_api['total_time_minutes']:.1f} minutes")
            print(f"   Cost: ${premium_api['total_cost']:.2f}")
            print(f"   Features: Multi-temperature, answer expansion")
        except ValueError:
            pass


def main():
    """Main calculator interface."""
    parser = argparse.ArgumentParser(
        description="Calculate operational characteristics for TruthfulQA benchmarks"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=817,
        help="Number of samples to evaluate (default: 817 = full validation set)"
    )
    parser.add_argument(
        "--provider", 
        choices=['openai', 'anthropic'],
        help="API provider for cost estimation"
    )
    parser.add_argument(
        "--model",
        help="Model name for cost estimation"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=1,
        help="Number of generations per sample"
    )
    parser.add_argument(
        "--temperatures",
        type=int, 
        default=1,
        help="Number of temperature variants"
    )
    parser.add_argument(
        "--expansion",
        action="store_true",
        help="Enable answer expansion"
    )
    parser.add_argument(
        "--reasoning",
        action="store_true", 
        help="Enable reasoning models"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Show comparison table of all methods"
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Show scaling analysis"
    )
    
    args = parser.parse_args()
    
    print("🧮 Coherify TruthfulQA Benchmark Calculator")
    print("=" * 50)
    
    print(f"\n📋 Dataset Information:")
    print(f"   TruthfulQA validation set: {TRUTHFULQA_VALIDATION_SIZE} samples")
    print(f"   Download time: {DOWNLOAD_TIME_SECONDS} seconds")
    print(f"   Dataset size: {DATASET_SIZE_MB} MB")
    print(f"   Average tokens per sample: {AVG_TOKENS_PER_SAMPLE}")
    
    # Specific calculation
    if args.provider and args.model:
        print(f"\n🔍 Detailed Analysis:")
        
        # Local processing
        local_result = calculate_local_processing(args.sample_size, 'hybrid')
        print(f"\n📱 Local Processing (HybridCoherence):")
        print(f"   Time: {local_result['total_time_minutes']:.1f} minutes")
        print(f"   Memory required: {local_result['memory_required_gb']} GB")
        print(f"   Throughput: {local_result['throughput_samples_per_minute']:.1f} samples/min")
        print(f"   Cost: Free")
        
        # API processing
        try:
            api_result = calculate_api_processing(
                args.sample_size,
                args.provider,
                args.model,
                args.generations,
                args.temperatures, 
                args.expansion,
                args.reasoning
            )
            
            print(f"\n🌐 API Processing ({args.provider}/{args.model}):")
            print(f"   Time: {api_result['total_time_minutes']:.1f} minutes")
            print(f"   API calls: {api_result['api_calls']} total ({api_result['api_calls_per_sample']:.1f} per sample)")
            print(f"   Input tokens: {api_result['total_input_tokens']:,}")
            print(f"   Output tokens: {api_result['total_output_tokens']:,}")
            print(f"   Cost: ${api_result['total_cost']:.3f} (${api_result['cost_per_sample']:.4f} per sample)")
            print(f"   Cost breakdown: ${api_result['input_cost']:.3f} input + ${api_result['output_cost']:.3f} output")
            
        except ValueError as e:
            print(f"   Error: {e}")
    
    # Comparison table
    if args.comparison:
        print_comparison_table(args.sample_size)
    
    # Scaling analysis
    if args.scaling:
        print_scaling_analysis()
    
    # Recommendations
    print_recommendations(args.sample_size)
    
    print(f"\n💡 Quick Commands:")
    print(f"   python scripts/benchmark_calculator.py --sample-size 50 --comparison")
    print(f"   python scripts/benchmark_calculator.py --provider openai --model gpt-4-turbo --generations 2")
    print(f"   python scripts/benchmark_calculator.py --scaling")


if __name__ == "__main__":
    main()
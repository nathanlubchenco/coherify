#!/usr/bin/env python3
"""
Diagnostic script to test benchmark fixes.

This script tests:
1. Config loading
2. Model initialization
3. Data loading
4. Basic evaluation
"""

import json
import os
import sys


def test_config():
    """Test config file loading."""
    print("=" * 60)
    print("1. Testing Config Loading")
    print("-" * 60)

    try:
        with open("config/benchmark_config.json") as f:
            config = json.load(f)
        print("✅ Config loaded successfully")

        # Check for model configs
        models = config.get("models", {})
        print(f"   Available models: {list(models.keys())}")

        # Check gpt4o config
        if "gpt4o" in models:
            print(f"   gpt4o config: {models['gpt4o']}")
        else:
            print("   ⚠️ gpt4o not found in config")

        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_api_key():
    """Test API key availability."""
    print("\n" + "=" * 60)
    print("2. Testing API Key Setup")
    print("-" * 60)

    key = os.getenv("OPENAI_API_KEY")
    if key:
        if key.startswith("sk-"):
            print(f"✅ OPENAI_API_KEY is set (starts with 'sk-')")
            return True
        else:
            print(f"⚠️ OPENAI_API_KEY is set but doesn't start with 'sk-'")
            return False
    else:
        print("❌ OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
        return False


def test_model_runner():
    """Test ModelRunner initialization."""
    print("\n" + "=" * 60)
    print("3. Testing ModelRunner")
    print("-" * 60)

    try:
        from coherify.generation.model_runner import ModelRunner

        # Load config
        with open("config/benchmark_config.json") as f:
            config = json.load(f)

        # Try to create ModelRunner
        model_config = config["models"]["gpt4o"]

        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ Skipping ModelRunner test (no API key)")
            return None

        runner = ModelRunner(model_config)
        print("✅ ModelRunner created successfully")

        # Test generation (small test)
        result = runner.generate_response("Say 'test'", temperature=0.1)
        if result and result.text:
            print(f"   Test generation successful: '{result.text[:50]}'")
            return True
        else:
            print("   ⚠️ Generation returned empty result")
            return False

    except Exception as e:
        print(f"❌ ModelRunner test failed: {e}")
        return False


def test_data_loading():
    """Test benchmark data loading."""
    print("\n" + "=" * 60)
    print("4. Testing Data Loading")
    print("-" * 60)

    results = {}

    # Test FEVER data
    print("\nFEVER Dataset:")
    try:
        from datasets import load_dataset

        # Try the alternative dataset
        dataset = load_dataset("copenlu/fever", split="validation", streaming=True)
        sample = next(iter(dataset))
        print(f"✅ FEVER data loads from copenlu/fever")
        print(f"   Sample fields: {list(sample.keys())}")
        results["fever"] = True
    except Exception as e:
        print(f"❌ FEVER data loading failed: {e}")
        results["fever"] = False

    # Test TruthfulQA data
    print("\nTruthfulQA Dataset:")
    try:
        dataset = load_dataset(
            "truthful_qa", "generation", split="validation", streaming=True
        )
        sample = next(iter(dataset))
        print(f"✅ TruthfulQA data loads")
        print(f"   Sample fields: {list(sample.keys())}")
        results["truthfulqa"] = True
    except Exception as e:
        print(f"❌ TruthfulQA data loading failed: {e}")
        results["truthfulqa"] = False

    return results


def test_benchmark_scripts():
    """Test if benchmark scripts run without errors."""
    print("\n" + "=" * 60)
    print("5. Testing Benchmark Scripts")
    print("-" * 60)

    scripts = [
        "examples/run_fever_benchmark.py",
        "examples/run_truthfulqa_benchmark.py",
        "examples/run_faithbench_benchmark.py",
    ]

    for script in scripts:
        print(f"\nTesting {script}:")
        if os.path.exists(script):
            # Try importing it
            try:
                import subprocess

                result = subprocess.run(
                    [sys.executable, script, "--sample-size", "1", "--model", "mock"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    print(f"✅ Script runs successfully")
                else:
                    print(f"⚠️ Script exited with code {result.returncode}")
                    if result.stderr:
                        print(f"   Error: {result.stderr[:200]}")
            except Exception as e:
                print(f"❌ Failed to run script: {e}")
        else:
            print(f"❌ Script not found")


def main():
    """Run all diagnostic tests."""
    print("🔍 Benchmark Diagnostics")
    print("=" * 60)

    results = {}

    # Run tests
    results["config"] = test_config()
    results["api_key"] = test_api_key()
    results["model_runner"] = test_model_runner()
    results["data"] = test_data_loading()
    test_benchmark_scripts()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)

    if results["config"]:
        print("✅ Config: OK")
    else:
        print("❌ Config: FAILED")

    if results["api_key"]:
        print("✅ API Key: OK")
    else:
        print("⚠️ API Key: NOT SET (benchmarks will use mock data)")

    if results["model_runner"]:
        print("✅ Model Runner: OK")
    elif results["model_runner"] is None:
        print("⚠️ Model Runner: SKIPPED (no API key)")
    else:
        print("❌ Model Runner: FAILED")

    if results.get("data"):
        data_results = results["data"]
        if data_results.get("fever"):
            print("✅ FEVER Data: OK")
        else:
            print("⚠️ FEVER Data: FAILED (will use mock data)")

        if data_results.get("truthfulqa"):
            print("✅ TruthfulQA Data: OK")
        else:
            print("⚠️ TruthfulQA Data: FAILED (will use mock data)")

    print("\n" + "=" * 60)
    print("💡 Next Steps:")
    print("-" * 60)

    if not results["api_key"]:
        print("1. Set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print()

    print("2. Run a benchmark with real model:")
    print("   make benchmark-fever MODEL=gpt4o SAMPLES=5")
    print()
    print("3. Compare all three stages:")
    print("   make benchmark-fever MODEL=gpt4o SAMPLES=10 COMPARE=true")
    print()
    print("4. Check results in results/ directory")


if __name__ == "__main__":
    main()

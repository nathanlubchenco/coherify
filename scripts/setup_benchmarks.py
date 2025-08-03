#!/usr/bin/env python3
"""
Setup script for Coherify benchmarks.

This script helps users set up the necessary dependencies and configuration
to run coherence benchmarks like TruthfulQA.
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_coherify_installation():
    """Check if coherify is properly installed."""
    try:
        import coherify
        print(f"✅ Coherify {coherify.__version__}")
        return True
    except ImportError:
        print("❌ Coherify not installed")
        print("   Install with: pip install -e .")
        return False


def install_optional_dependencies():
    """Install optional dependencies for benchmarks."""
    print("\n📦 Installing benchmark dependencies...")
    
    # Core benchmark dependencies
    dependencies = [
        "datasets>=2.0.0",
        "requests>=2.25.0",
        "evaluate>=0.3.0"
    ]
    
    # API dependencies
    api_dependencies = [
        "openai>=1.0.0",
        "anthropic>=0.8.0"
    ]
    
    # Visualization dependencies
    viz_dependencies = [
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0"
    ]
    
    for dep_group, deps in [
        ("Core", dependencies),
        ("API", api_dependencies), 
        ("Visualization", viz_dependencies)
    ]:
        print(f"\n  Installing {dep_group} dependencies...")
        
        for dep in deps:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", dep
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"    ✅ {dep}")
            except subprocess.CalledProcessError:
                print(f"    ❌ {dep} (failed to install)")


def check_api_keys():
    """Check for API keys and help user set them up."""
    print("\n🔑 Checking API keys...")
    
    api_keys = {
        "OPENAI_API_KEY": "OpenAI API key",
        "ANTHROPIC_API_KEY": "Anthropic API key"
    }
    
    found_keys = []
    missing_keys = []
    
    for env_var, description in api_keys.items():
        if os.getenv(env_var):
            print(f"✅ {description}")
            found_keys.append(env_var)
        else:
            print(f"❌ {description}")
            missing_keys.append(env_var)
    
    if missing_keys:
        print(f"\n💡 To set up API keys:")
        for key in missing_keys:
            print(f"   export {key}='your-api-key-here'")
        
        print(f"\n   Or add to your shell profile (~/.bashrc, ~/.zshrc):")
        for key in missing_keys:
            print(f"   echo 'export {key}=\"your-api-key-here\"' >> ~/.bashrc")
    
    return len(found_keys) > 0


def create_example_config():
    """Create example configuration files."""
    print("\n📝 Creating example configuration...")
    
    # Create config directory
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Example benchmark config
    benchmark_config = {
        "truthfulqa": {
            "sample_size": 50,
            "evaluation_mode": "generation",
            "use_api_enhancement": True,
            "coherence_measures": [
                "SemanticCoherence",
                "HybridCoherence",
                "APIEnhancedHybridCoherence"
            ]
        },
        "api_settings": {
            "use_temperature_variance": True,
            "temperature_range": [0.3, 0.7, 1.0],
            "num_generations_per_prompt": 2,
            "enable_confidence_scoring": True
        }
    }
    
    config_file = config_dir / "benchmark_config.json"
    with open(config_file, "w") as f:
        json.dump(benchmark_config, f, indent=2)
    
    print(f"✅ Created {config_file}")
    
    # Example environment file
    env_file = config_dir / "example.env"
    with open(env_file, "w") as f:
        f.write("# Copy this file to .env and fill in your API keys\n")
        f.write("OPENAI_API_KEY=your_openai_key_here\n")
        f.write("ANTHROPIC_API_KEY=your_anthropic_key_here\n")
        f.write("\n# Optional: Organization ID for OpenAI\n")
        f.write("# OPENAI_ORG_ID=your_org_id\n")
        f.write("\n# Optional: Base URLs for API endpoints\n")
        f.write("# OPENAI_BASE_URL=https://api.openai.com/v1\n")
        f.write("# ANTHROPIC_BASE_URL=https://api.anthropic.com\n")
    
    print(f"✅ Created {env_file}")


def verify_setup():
    """Verify that everything is set up correctly."""
    print("\n🔍 Verifying setup...")
    
    # Test basic import
    try:
        from coherify import PropositionSet, HybridCoherence
        print("✅ Core coherify imports work")
    except ImportError as e:
        print(f"❌ Core imports failed: {e}")
        return False
    
    # Test benchmark imports
    try:
        from coherify import TruthfulQAAdapter, TruthfulQAEvaluator
        print("✅ Benchmark imports work")
    except ImportError as e:
        print(f"❌ Benchmark imports failed: {e}")
    
    # Test API imports
    try:
        from coherify.measures.api_enhanced import APIEnhancedHybridCoherence
        from coherify.benchmarks.api_enhanced import APIEnhancedQAAdapter
        print("✅ API-enhanced imports work")
    except ImportError as e:
        print(f"⚠️  API-enhanced imports failed: {e}")
    
    # Test datasets library
    try:
        import datasets
        print("✅ datasets library available")
    except ImportError:
        print("⚠️  datasets library not available")
    
    # Test basic functionality
    try:
        prop_set = PropositionSet.from_qa_pair(
            "Test question?",
            "This is a test answer to verify functionality."
        )
        measure = HybridCoherence()
        result = measure.compute(prop_set)
        print(f"✅ Basic functionality test passed (score: {result.score:.3f})")
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False
    
    return True


def show_next_steps():
    """Show what the user should do next."""
    print("\n🎯 Next Steps:")
    print("\n1. Run a basic benchmark:")
    print("   python examples/run_truthfulqa_benchmark.py --sample-size 5")
    
    print("\n2. Run with API enhancement (requires API keys):")
    print("   python examples/run_truthfulqa_benchmark.py --use-api --sample-size 3")
    
    print("\n3. Run other examples:")
    print("   python examples/basic_usage.py")
    print("   python examples/practical_applications.py")
    
    print("\n4. Explore the codebase:")
    print("   - coherify/measures/         # Coherence measure implementations")
    print("   - coherify/benchmarks/       # Benchmark adapters")
    print("   - coherify/providers/        # API provider integrations")
    print("   - examples/                  # Usage examples")
    
    print("\n5. Read the documentation:")
    print("   - README.md                  # Main documentation")
    print("   - examples/                  # Practical examples")


def main():
    """Main setup function."""
    print("🚀 Coherify Benchmark Setup")
    print("=" * 40)
    
    # Check requirements
    if not check_python_version():
        return
    
    if not check_coherify_installation():
        return
    
    # Install dependencies
    install_optional_dependencies()
    
    # Check API setup
    has_api_keys = check_api_keys()
    
    # Create configuration
    create_example_config()
    
    # Verify setup
    if verify_setup():
        print("\n✅ Setup completed successfully!")
        
        if has_api_keys:
            print("🎉 Ready for API-enhanced benchmarks!")
        else:
            print("⚠️  Set up API keys for enhanced features")
        
        show_next_steps()
    else:
        print("\n❌ Setup verification failed")
        print("   Please check the error messages above")


if __name__ == "__main__":
    main()
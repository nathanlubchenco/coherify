#!/usr/bin/env python3
"""
Check if all required dependencies are properly installed.
"""

import sys
import importlib
from typing import List, Tuple

def check_imports() -> Tuple[List[str], List[str]]:
    """Check if all required packages can be imported."""
    
    required_packages = {
        # Core dependencies
        "numpy": "numpy",
        "torch": "torch", 
        "transformers": "transformers",
        "sentence_transformers": "sentence-transformers",
        "datasets": "datasets",
        "sklearn": "scikit-learn",
        "pandas": "pandas",
        
        # API providers
        "openai": "openai",
        "anthropic": "anthropic",
        
        # Utilities
        "tqdm": "tqdm",
        "yaml": "pyyaml",
        "requests": "requests",
        
        # Optional but useful
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "streamlit": "streamlit",
        "plotly": "plotly",
        
        # NLP utilities
        "huggingface_hub": "huggingface-hub",
        "tokenizers": "tokenizers",
    }
    
    success = []
    failed = []
    
    print("🔍 Checking Coherify Dependencies")
    print("=" * 50)
    
    for import_name, package_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            success.append(package_name)
            print(f"✅ {package_name:25} installed")
        except ImportError as e:
            failed.append(package_name)
            print(f"❌ {package_name:25} NOT installed")
    
    return success, failed


def check_coherify_modules():
    """Check if Coherify modules can be imported."""
    
    print("\n🔍 Checking Coherify Modules")
    print("=" * 50)
    
    modules = [
        "coherify.core.base",
        "coherify.measures.semantic",
        "coherify.benchmarks.adapters",
        "coherify.providers.openai_provider",
        "coherify.generation.model_runner",
        "coherify.evaluators.response_selectors",
    ]
    
    all_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Main function."""
    
    # Check external dependencies
    success, failed = check_imports()
    
    # Check Coherify modules
    coherify_ok = check_coherify_modules()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary")
    print("=" * 50)
    
    if failed:
        print(f"\n❌ Missing {len(failed)} dependencies:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("\n💡 To install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("\n💡 Or install just the missing ones:")
        for pkg in failed:
            print(f"   pip install {pkg}")
    else:
        print("✅ All external dependencies installed!")
    
    if not coherify_ok:
        print("\n⚠️  Some Coherify modules failed to import")
        print("   Try: pip install -e .")
    else:
        print("✅ All Coherify modules working!")
    
    # Return appropriate exit code
    if failed or not coherify_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Verify the Coherify environment is properly set up.
"""

import sys
import os
from pathlib import Path

def check_environment():
    """Check environment setup."""
    
    print("🔍 Environment Verification")
    print("=" * 60)
    
    # Python version
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # Virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    print(f"Virtual Environment: {'✅ Active' if in_venv else '❌ Not active'}")
    
    if in_venv:
        print(f"Virtual Env Path: {sys.prefix}")
    
    # Coherify installation
    print(f"\nCoherify Installation:")
    try:
        import coherify
        print(f"✅ Coherify found at: {coherify.__file__}")
        
        # Check if it's editable install
        coherify_path = Path(coherify.__file__).parent.parent
        if coherify_path == Path.cwd():
            print("✅ Editable installation (development mode)")
        else:
            print("⚠️  Not an editable installation")
            
    except ImportError:
        print("❌ Coherify not installed")
        print("   Run: pip install -e .")
    
    # API Keys
    print(f"\nAPI Keys:")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    print(f"OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not set'}")
    print(f"ANTHROPIC_API_KEY: {'✅ Set' if anthropic_key else '❌ Not set'}")
    
    if not openai_key:
        print("   To set: export OPENAI_API_KEY=your-key-here")
    
    # Critical imports test
    print(f"\nCritical Imports Test:")
    critical = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("openai", "OpenAI"),
    ]
    
    all_ok = True
    for module, name in critical:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Run: pip install {module.replace('_', '-')}")
            all_ok = False
    
    # Quick functionality test
    print(f"\nFunctionality Test:")
    try:
        from coherify.core.base import Proposition, PropositionSet
        from coherify.measures.semantic import SemanticCoherence
        
        # Create simple test
        props = [
            Proposition(text="The sky is blue"),
            Proposition(text="The ocean is blue")
        ]
        prop_set = PropositionSet(propositions=props)
        
        measure = SemanticCoherence()
        result = measure.compute(prop_set)
        
        print(f"✅ Basic coherence computation works")
        print(f"   Test coherence score: {result.score:.3f}")
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok and in_venv and openai_key:
        print("✅ Environment is properly configured!")
        print("\nYou can now run:")
        print("  python examples/run_full_pipeline_comparison.py --model gpt4-mini --samples 5")
    else:
        print("⚠️  Some issues detected. See above for details.")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(check_environment())
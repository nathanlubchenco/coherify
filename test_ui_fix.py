#!/usr/bin/env python3
"""
Test script to verify the UI fixes are working correctly.
"""

import sys
from pathlib import Path

# Add coherify to path
sys.path.append(str(Path(__file__).parent))

from coherify.core.base import Proposition, PropositionSet
from ui.coherence_app import compute_coherence_scores, create_coherence_measures, create_pairwise_heatmap
from coherify.utils.clean_output import enable_clean_output

enable_clean_output()

def test_ui_functionality():
    """Test that the UI functions work without errors."""
    
    print("🧪 Testing UI Fix - Pairwise Scores")
    print("=" * 40)
    
    # Create test propositions
    propositions = [
        "Global temperatures have risen by 1.1°C since pre-industrial times.",
        "The increase is primarily due to greenhouse gas emissions.",
        "Climate scientists agree that immediate action is needed."
    ]
    
    prop_objects = [Proposition(text=prop) for prop in propositions]
    prop_set = PropositionSet(propositions=prop_objects, context="Climate change")
    
    # Get available measures
    measures = create_coherence_measures()
    print(f"📊 Available measures: {list(measures.keys())}")
    
    # Mock streamlit progress components
    class MockProgress:
        def progress(self, value): pass
        def empty(self): pass
        def text(self, text): print(f"  {text}")
    
    class MockStreamlit:
        def progress(self, value): return MockProgress()
        def empty(self): return MockProgress()
        def warning(self, msg): print(f"⚠️  {msg}")
    
    # Monkey patch for testing
    import ui.coherence_app as app
    original_st = app.st
    app.st = MockStreamlit()
    
    try:
        # Test coherence computation
        print("\n🔍 Computing coherence scores...")
        results = compute_coherence_scores(prop_set, measures)
        
        print("\n📊 Results:")
        for name, data in results.items():
            if "error" not in data:
                pairwise_count = len(data.get("pairwise_scores", []))
                print(f"  {name:>20}: {data['score']:.3f} (pairwise: {pairwise_count})")
            else:
                print(f"  {name:>20}: ERROR - {data['error']}")
        
        # Test visualization creation
        print("\n📈 Testing heatmap creation...")
        heatmap_fig = create_pairwise_heatmap(results, propositions)
        if heatmap_fig:
            print("  ✅ Heatmap created successfully")
        else:
            print("  ℹ️  No heatmap data available (expected for some measures)")
        
        print("\n✅ UI functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original streamlit
        app.st = original_st

if __name__ == "__main__":
    success = test_ui_functionality()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Demo script showing the Coherify UI functionality without launching Streamlit.
This demonstrates the core coherence analysis features.
"""

import sys
from pathlib import Path

# Add coherify to path
sys.path.append(str(Path(__file__).parent))

from coherify.core.base import Proposition, PropositionSet
from coherify.measures.semantic import SemanticCoherence
from coherify.measures.entailment import EntailmentCoherence
from coherify.measures.hybrid import HybridCoherence
from coherify.utils.clean_output import enable_clean_output

# Enable clean output
enable_clean_output()

def demo_coherence_analysis():
    """Demonstrate coherence analysis on different examples."""
    
    print("🧠 Coherify UI Demo - Coherence Analysis")
    print("=" * 50)
    
    # Example texts from the UI
    examples = {
        "Coherent News Story": [
            "Global temperatures have risen by 1.1°C since pre-industrial times.",
            "The increase is primarily due to greenhouse gas emissions.",
            "Climate scientists agree that immediate action is needed.",
            "Renewable energy adoption is accelerating worldwide."
        ],
        "Incoherent Mixed Topics": [
            "The sky is blue on a clear day.",
            "Pizza was invented in Italy.",
            "Quantum computers use quantum bits called qubits.",
            "My favorite color is purple."
        ],
        "Contradictory Statements": [
            "Today is sunny and bright.",
            "It's raining heavily outside.",
            "The weather is perfect for a picnic.",
            "Everyone should stay indoors due to the storm."
        ]
    }
    
    # Initialize measures
    measures = {
        "Semantic": SemanticCoherence(),
        "Entailment": EntailmentCoherence(),
        "Hybrid": HybridCoherence()
    }
    
    # Analyze each example
    for example_name, propositions in examples.items():
        print(f"\n📝 Analyzing: {example_name}")
        print("-" * 30)
        
        # Display propositions
        for i, prop in enumerate(propositions, 1):
            print(f"  {i}. {prop}")
        
        # Create proposition set
        prop_objects = [Proposition(text=prop) for prop in propositions]
        prop_set = PropositionSet(propositions=prop_objects, context=example_name)
        
        print(f"\n📊 Coherence Scores:")
        
        # Compute coherence for each measure
        for measure_name, measure in measures.items():
            try:
                result = measure.compute(prop_set)
                print(f"  {measure_name:>12}: {result.score:.3f}")
                if hasattr(result, 'pairwise_scores') and result.pairwise_scores:
                    avg_pairwise = sum(result.pairwise_scores) / len(result.pairwise_scores)
                    print(f"  {'':>12}  (avg pairwise: {avg_pairwise:.3f})")
            except Exception as e:
                print(f"  {measure_name:>12}: Error - {str(e)[:50]}...")
        
        print()

def show_ui_features():
    """Show what features are available in the UI."""
    
    print("\n🎨 Coherify Interactive UI Features")
    print("=" * 40)
    
    features = [
        "🔬 Interactive Coherence Workbench",
        "   • Text input for custom propositions",
        "   • Real-time coherence analysis",
        "   • Multiple coherence measures comparison",
        "   • Predefined examples and presets",
        "",
        "📊 Visual Analysis Tools",
        "   • Bar charts comparing coherence scores",
        "   • Pairwise similarity heatmaps",
        "   • Interactive plotly visualizations",
        "",
        "📈 Benchmark Performance Dashboard",
        "   • FEVER, TruthfulQA, FaithBench results",
        "   • Baseline vs coherence-guided comparison",
        "   • Performance improvement metrics",
        "",
        "🎓 Educational Features",
        "   • Explanations of each coherence measure",
        "   • Interactive examples and tutorials",
        "   • Help text and tooltips throughout"
    ]
    
    for feature in features:
        print(feature)
    
    print(f"\n🌐 To launch the full interactive UI:")
    print(f"   python run_ui.py")
    print(f"   # or")
    print(f"   make ui")
    print(f"\n📍 The UI will be available at: http://localhost:8501")

if __name__ == "__main__":
    try:
        demo_coherence_analysis()
        show_ui_features()
        
        print("\n✨ Demo completed successfully!")
        print("Launch the full UI to explore interactively!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
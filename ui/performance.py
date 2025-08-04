"""
Performance optimization utilities for the Coherify UI.
"""

import streamlit as st
import time
from functools import lru_cache
from typing import Dict, Any, Optional
import threading
import queue


class ModelManager:
    """Singleton class to manage model loading and caching."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self._models = {}
            self._loading = set()
            self._initialized = True
    
    def get_measure(self, measure_name: str, measure_class, *args, **kwargs):
        """Get or create a coherence measure with caching."""
        if measure_name not in self._models:
            if measure_name not in self._loading:
                self._loading.add(measure_name)
                try:
                    self._models[measure_name] = measure_class(*args, **kwargs)
                except Exception as e:
                    # Handle case where streamlit is not available
                    try:
                        import streamlit as st
                        st.error(f"Failed to load {measure_name}: {str(e)}")
                    except:
                        print(f"Failed to load {measure_name}: {str(e)}")
                    return None
                finally:
                    self._loading.discard(measure_name)
        
        return self._models.get(measure_name)
    
    def is_loading(self, measure_name: str) -> bool:
        """Check if a measure is currently loading."""
        return measure_name in self._loading


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_example_texts():
    """Cache example texts to avoid recreation."""
    return {
        "Custom": {
            "context": "", 
            "propositions": [], 
            "description": "Enter your own text"
        },
        "Coherent News Story": {
            "context": "Breaking news about climate change",
            "propositions": [
                "Global temperatures have risen by 1.1°C since pre-industrial times.",
                "The increase is primarily due to greenhouse gas emissions.",
                "Climate scientists agree that immediate action is needed.",
                "Renewable energy adoption is accelerating worldwide."
            ],
            "description": "Example of coherent, related statements about a single topic."
        },
        "Incoherent Mixed Topics": {
            "context": "Random facts",
            "propositions": [
                "The sky is blue on a clear day.",
                "Pizza was invented in Italy.", 
                "Quantum computers use quantum bits called qubits.",
                "My favorite color is purple."
            ],
            "description": "Unrelated statements that should score low on coherence."
        },
        "Contradictory Statements": {
            "context": "Weather conditions", 
            "propositions": [
                "Today is sunny and bright.",
                "It's raining heavily outside.",
                "The weather is perfect for a picnic.",
                "Everyone should stay indoors due to the storm."
            ],
            "description": "Contradictory statements detected by entailment measures."
        }
    }


@st.cache_data(ttl=60)  # Cache results for 1 minute
def compute_coherence_cached(propositions_hash: str, measures_selected: tuple):
    """Cache coherence computation results."""
    # This will be populated by the main compute function
    return None


def optimize_streamlit_config():
    """Configure Streamlit for better performance."""
    # Disable expensive features
    st.set_page_config(
        page_title="Coherify - Coherence Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )


class ProgressTracker:
    """Track and display progress for long operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.start_time = time.time()
    
    def update(self, step_name: str = ""):
        """Update progress."""
        self.current_step += 1
        progress = self.current_step / self.total_steps
        self.progress_bar.progress(progress)
        
        elapsed = time.time() - self.start_time
        if self.current_step > 0:
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            self.status_text.text(f"{self.description}: {step_name} (ETA: {eta:.1f}s)")
        else:
            self.status_text.text(f"{self.description}: {step_name}")
    
    def complete(self):
        """Mark as complete and clean up."""
        self.progress_bar.empty()
        self.status_text.empty()


def get_fast_measures():
    """Get measures optimized for speed."""
    from coherify.measures.semantic import SemanticCoherence
    from coherify.measures.hybrid import HybridCoherence
    
    manager = ModelManager()
    
    # Use smaller, faster models
    fast_measures = {}
    
    # Fast semantic measure with default encoder (fallback to built-in)
    try:
        # Try with custom encoder first
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            semantic = manager.get_measure(
                "fast_semantic", 
                SemanticCoherence,
                encoder=encoder,
            )
        except ImportError:
            # Fallback to default encoder
            semantic = manager.get_measure(
                "fast_semantic", 
                SemanticCoherence,
            )
    except Exception as e:
        print(f"Could not load fast semantic measure: {e}")
        semantic = None
    if semantic:
        fast_measures["Semantic Coherence"] = {
            "measure": semantic,
            "description": "Fast embedding-based similarity analysis",
            "color": "#2E86AB"  # Professional blue
        }
    
    # Hybrid measure (creates its own internal measures)
    if semantic:
        try:
            # HybridCoherence creates its own semantic and entailment measures internally
            hybrid = manager.get_measure(
                "fast_hybrid",
                HybridCoherence,
                semantic_weight=0.6,  # Favor semantic for fast mode
                entailment_weight=0.4,
            )
            if hybrid:
                fast_measures["Hybrid Coherence"] = {
                    "measure": hybrid,
                    "description": "Balanced semantic and logical analysis",
                    "color": "#A23B72"  # Professional purple
                }
        except Exception as e:
            print(f"Could not load hybrid measure: {e}")
    
    return fast_measures


def get_slow_measures():
    """Get measures that are slower but more comprehensive."""
    from coherify.measures.entailment import EntailmentCoherence
    
    manager = ModelManager()
    slow_measures = {}
    
    # Entailment measure (slower due to NLI model)
    entailment = manager.get_measure(
        "entailment",
        EntailmentCoherence
    )
    if entailment:
        slow_measures["Entailment Coherence"] = {
            "measure": entailment,
            "description": "Logical relationship analysis using NLI",
            "color": "#F18F01"  # Professional orange
        }
    
    return slow_measures


def get_advanced_measures():
    """Get advanced measures including Shogenji and API-enhanced options."""
    from coherify.measures.shogenji import ShogunjiCoherence
    
    manager = ModelManager()
    advanced_measures = {}
    
    # Improved Shogenji measure with better probability estimation
    try:
        from coherify.measures.shogenji import ConfidenceBasedProbabilityEstimator
        
        # Create improved probability estimator
        improved_estimator = ConfidenceBasedProbabilityEstimator(
            baseline_prob=0.5,
            prob_range=(0.2, 0.8)
        )
        
        shogenji = manager.get_measure(
            "improved_shogenji",
            ShogunjiCoherence,
            probability_estimator=improved_estimator
        )
        if shogenji:
            advanced_measures["Shogenji Coherence"] = {
                "measure": shogenji,
                "description": "Traditional probability-based coherence measure from philosophy. Unbounded values represent the degree to which propositions are more coherent together than if independent.",
                "color": "#9467BD",  # Professional purple
                "unbounded": True,  # Flag to indicate special handling
                "explanation": "Shogenji coherence C_S = P(H₁∧H₂∧...∧Hₙ) / ∏P(Hᵢ) measures how much more likely propositions are together vs independently. Values >1 indicate positive coherence, =1 indicates independence, <1 indicates incoherence. High values suggest strong mutual support between propositions."
            }
    except Exception as e:
        print(f"Could not load Shogenji measure: {e}")
    
    return advanced_measures


def get_api_enhanced_measures(api_provider=None, model_name=None):
    """Get API-enhanced coherence measures."""
    api_measures = {}
    
    if not api_provider or not model_name:
        return api_measures
    
    try:
        # Check if API-enhanced coherence is available
        try:
            from coherify.measures.api_enhanced import APIEnhancedCoherence
            
            manager = ModelManager()
            
            # API-enhanced coherence measure
            api_measure = manager.get_measure(
                f"api_{api_provider}_{model_name}",
                APIEnhancedCoherence,
                provider=api_provider,
                model=model_name
            )
            
            if api_measure:
                api_measures[f"{api_provider.title()} Enhanced"] = {
                    "measure": api_measure,
                    "description": f"AI-enhanced coherence using {api_provider} {model_name}",
                    "color": "#E74C3C" if api_provider == "anthropic" else "#3498DB"
                }
        
        except ImportError:
            # API-enhanced coherence not available - this is expected for now
            print(f"Note: API-enhanced coherence not yet implemented")
            # For now, return a placeholder that uses the standard measures with API context
            pass
        
    except Exception as e:
        print(f"Could not load API-enhanced measure: {e}")
    
    return api_measures
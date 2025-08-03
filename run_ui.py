#!/usr/bin/env python3
"""
Launch the Coherify Interactive UI

This script launches the Streamlit-based web interface for exploring
coherence measures and benchmark performance.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    # Get the path to the UI app
    ui_path = Path(__file__).parent / "ui" / "coherence_app.py"
    
    if not ui_path.exists():
        print(f"Error: UI app not found at {ui_path}")
        sys.exit(1)
    
    print("🧠 Launching Coherify Interactive UI...")
    print(f"📍 Starting Streamlit server...")
    print(f"🌐 The app will open in your browser at http://localhost:8501")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(ui_path),
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Coherify UI...")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
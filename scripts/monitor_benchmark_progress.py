#!/usr/bin/env python3
"""
Monitor benchmark progress by checking files and showing real-time status.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

def monitor_progress():
    """Monitor benchmark progress by checking output files."""
    
    results_dir = Path("results")
    
    print("📊 Monitoring Benchmark Progress")
    print("=" * 60)
    print("Press Ctrl+C to stop monitoring\n")
    
    last_file_count = 0
    last_check = time.time()
    
    while True:
        try:
            # Check for result files
            if results_dir.exists():
                json_files = list(results_dir.glob("*.json"))
                
                if len(json_files) != last_file_count:
                    print(f"\n🆕 New result files detected: {len(json_files)} total")
                    last_file_count = len(json_files)
                    
                    # Show recent files
                    recent_files = sorted(json_files, key=lambda x: x.stat().st_mtime)[-3:]
                    for f in recent_files:
                        mtime = datetime.fromtimestamp(f.stat().st_mtime)
                        print(f"  📄 {f.name} - {mtime.strftime('%H:%M:%S')}")
                        
                        # Try to read and show progress
                        try:
                            with open(f) as jf:
                                data = json.load(jf)
                                if "evaluation_config" in data:
                                    config = data["evaluation_config"]
                                    print(f"     Samples: {config.get('sample_size', 'N/A')}")
                                if "native_metrics" in data:
                                    metrics = data["native_metrics"]
                                    print(f"     Truthful: {metrics.get('truthful_score', 0):.1%}")
                        except:
                            pass
            
            # Check process status
            current_time = time.time()
            if current_time - last_check > 10:
                print(f"\r⏱️  Status at {datetime.now().strftime('%H:%M:%S')} - Files: {last_file_count}", end="", flush=True)
                last_check = current_time
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_progress()
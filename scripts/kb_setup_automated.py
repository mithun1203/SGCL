"""
Automated Knowledge Base Setup for Kaggle/Colab
================================================

This script automatically downloads and sets up the ConceptNet KB
without any user prompts - perfect for Kaggle notebooks.

Usage:
    python scripts/kb_setup_automated.py
"""

import sys
import subprocess
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DOWNLOAD_SCRIPT = SCRIPT_DIR / "download_full_conceptnet.py"
KB_PATH = PROJECT_ROOT / "sid" / "knowledge_base.json"

def main():
    print("=" * 70)
    print("ConceptNet KB Automated Setup")
    print("=" * 70)
    print()
    
    # Check if KB already exists
    if KB_PATH.exists():
        import json
        try:
            with open(KB_PATH, 'r', encoding='utf-8') as f:
                kb = json.load(f)
            concept_count = len(kb)
            print(f"KB already exists: {concept_count:,} concepts")
            
            if concept_count > 100000:
                print("KB looks good - skipping download")
                return 0
            else:
                print("KB seems incomplete - will re-download")
        except Exception as e:
            print(f"KB file corrupted: {e}")
            print("  Will re-download...")
    
    # Run download script with --auto flag
    print()
    print("Starting automated KB download...")
    print("This will take 2-4 hours (download + processing)")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, str(DOWNLOAD_SCRIPT), "--auto"],
            check=False,
            cwd=PROJECT_ROOT
        )
        
        if result.returncode != 0:
            print()
            print("KB download failed")
            return 1
        
        # Verify the download
        if KB_PATH.exists():
            import json
            with open(KB_PATH, 'r', encoding='utf-8') as f:
                kb = json.load(f)
            concept_count = len(kb)
            print()
            print("=" * 70)
            print(f"SUCCESS: KB ready with {concept_count:,} concepts")
            print("=" * 70)
            return 0
        else:
            print()
            print("KB file not found after download")
            return 1
            
    except Exception as e:
        print()
        print(f"Error during KB setup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

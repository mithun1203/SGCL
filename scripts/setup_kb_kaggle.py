"""
Setup full ConceptNet KB for Kaggle environment.
Run this once before training to download the complete KB.
"""
import subprocess
import sys

print(" Setting up full ConceptNet KB for offline operation...")
print("=" * 60)

# Run the download script with --auto flag (skips interactive prompt)
print("\n Downloading ConceptNet 5.7...")
result = subprocess.run([sys.executable, "scripts/download_full_conceptnet.py", "--auto"], 
                       capture_output=False)

if result.returncode == 0:
    print("\n KB Setup Complete!")
    
    # Verify
    import json
    kb = json.load(open('sid/knowledge_base.json', encoding='utf-8'))
    print(f"\n KB Statistics:")
    print(f"   - Concepts: {len(kb['concepts']):,}")
    print(f"   - Total edges: {len(kb['edges']):,}")
    print(f"   - English edges: {sum(1 for e in kb['edges'] if e.get('lang')=='en'):,}")
    print(f"   - Offline mode: ENABLED")
    print(f"\n Ready to train with full KB!")
else:
    print("\n KB download failed!")
    sys.exit(1)

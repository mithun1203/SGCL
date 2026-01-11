"""
Automatic setup of full ConceptNet KB for Kaggle.
No user interaction. Safe for notebooks.
"""

import subprocess
import sys
import json
from pathlib import Path

print("🚀 Setting up full ConceptNet KB for offline operation...")
print("=" * 60)

SCRIPT_PATH = Path("scripts/download_full_conceptnet.py")

if not SCRIPT_PATH.exists():
    print("❌ Download script not found:", SCRIPT_PATH)
    sys.exit(1)

print("\n📦 Downloading ConceptNet 5.7 (automatic mode)...")

result = subprocess.run(
    [sys.executable, str(SCRIPT_PATH), "--auto"],
    check=False
)

if result.returncode != 0:
    print("\n❌ KB download failed.")
    sys.exit(1)

print("\n✅ KB Download complete.")

# Verify installation
try:
    with open("sid/knowledge_base.json", encoding="utf-8") as f:
        kb = json.load(f)

    print("\n📊 KB Statistics:")
    print(f"   • Concepts     : {len(kb['concepts']):,}")
    print(f"   • Total edges  : {len(kb['edges']):,}")
    print(f"   • English edges: {sum(1 for e in kb['edges'] if e.get('lang') == 'en'):,}")
    print("   • Offline mode : ENABLED")

    print("\n🎉 ConceptNet KB is ready. Training can start.")
except Exception as e:
    print("\n❌ KB verification failed:", e)
    sys.exit(1)

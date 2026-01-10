"""
Emergency Fix: Enable SID Offline Mode
=======================================

Run this BEFORE training to switch SID to offline mode when ConceptNet API is down.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Monkey-patch SID to use offline mode
def patch_sid_offline():
    """Patch SemanticInconsistencyDetector to use offline KB only."""
    from sid.conceptnet_client import ConceptNetConfig
    
    # Save original __init__
    original_init = ConceptNetConfig.__init__
    
    def patched_init(self, **kwargs):
        # Force offline mode
        kwargs['offline_only'] = True
        kwargs['knowledge_base_path'] = str(Path(__file__).parent.parent / "sid" / "knowledge_base.json")
        original_init(self, **kwargs)
    
    ConceptNetConfig.__init__ = patched_init
    print("✓ SID patched to offline mode")
    print(f"  Using local KB: sid/knowledge_base.json")

if __name__ == "__main__":
    patch_sid_offline()
    
    # Test it works
    from sid.detector import SemanticInconsistencyDetector
    
    print("\nTesting SID in offline mode...")
    sid = SemanticInconsistencyDetector()
    result = sid.detect_conflict("Birds can fly. Penguins cannot fly.")
    
    if result.has_conflict:
        print(f"✓ Offline SID working! Detected conflict.")
    else:
        print("⚠️  No conflict detected - check KB")

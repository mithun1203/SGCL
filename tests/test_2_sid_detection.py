"""
Test 2: SID Conflict Detection

Verifies:
- Detects contradiction
- Detects hierarchy conflict
- No false positives on neutral statements
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sid.detector import SemanticInconsistencyDetector

def test_sid_detection():
    """Test SID conflict detection capabilities."""
    
    print("="*70)
    print("TEST 2: SID Conflict Detection")
    print("="*70)
    
    print("\n📦 Initializing SID (offline mode)...")
    sid = SemanticInconsistencyDetector()
    print("✓ SID initialized successfully")
    
    # Test cases: (text, expected_conflict, description)
    test_cases = [
        (
            "Birds can fly. Penguins cannot fly.",
            True,
            "Exception violation (penguins are birds but can't fly)"
        ),
        (
            "A dog is an animal. A dog is not an animal.",
            True,
            "Direct contradiction"
        ),
        (
            "The sky is blue.",
            False,
            "Neutral statement (no conflict)"
        ),
        (
            "Mammals are warm-blooded. Whales are mammals.",
            False,
            "Consistent facts (no conflict)"
        ),
        (
            "All birds have feathers. Penguins are birds. Penguins cannot fly.",
            True,
            "Complex multi-statement conflict"
        )
    ]
    
    print("\n" + "-"*70)
    print("Running Test Cases")
    print("-"*70)
    
    passed = 0
    failed = 0
    
    for i, (text, expected_conflict, description) in enumerate(test_cases, 1):
        print(f"\nTest 2.{i}: {description}")
        print(f"Input: \"{text}\"")
        
        try:
            result = sid.detect_conflict(text)
            has_conflict = result.has_conflict
            
            print(f"Expected conflict: {expected_conflict}")
            print(f"Detected conflict: {has_conflict}")
            
            if has_conflict == expected_conflict:
                print(f"✅ PASS")
                if has_conflict and hasattr(result, 'primary_type'):
                    print(f"   Conflict type: {result.primary_type}")
                if hasattr(result, 'confidence'):
                    print(f"   Confidence: {result.confidence:.2f}")
                if hasattr(result, 'detected_conflicts') and result.detected_conflicts:
                    print(f"   Detected conflicts: {len(result.detected_conflicts)}")
                passed += 1
            else:
                print(f"❌ FAIL: Expected {expected_conflict}, got {has_conflict}")
                failed += 1
                
        except Exception as e:
            print(f"❌ FAIL: Exception occurred: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    if failed == 0:
        print("✅ ALL SID TESTS PASSED")
        print("="*70)
        print(f"Tests run: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Pass rate: {pass_rate:.1f}%")
        print("="*70)
        return True
    else:
        print("❌ SOME SID TESTS FAILED")
        print("="*70)
        print(f"Tests run: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Pass rate: {pass_rate:.1f}%")
        print("="*70)
        return False


if __name__ == "__main__":
    try:
        success = test_sid_detection()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

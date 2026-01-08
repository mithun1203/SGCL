"""
Pre-Integration Test Suite - Run All Tests

Executes all 5 pre-integration tests in sequence.
"""

import subprocess
import sys
from pathlib import Path

def run_test(test_num: int, test_name: str) -> bool:
    """Run a single test and return success status."""
    print("\n" + "="*70)
    print(f"Running Test {test_num}: {test_name}")
    print("="*70)
    
    test_file = Path(__file__).parent / f"test_{test_num}_{test_name}.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0


def main():
    """Run all pre-integration tests."""
    print("="*70)
    print("SG-CL PRE-INTEGRATION TEST SUITE")
    print("="*70)
    print("\nRunning all component tests before integration...")
    
    tests = [
        (1, "seca_dataset", "SeCA Dataset Validation"),
        (2, "sid_detection", "SID Conflict Detection"),
        (3, "guardrail_generator", "Guard-Rail Generator"),
    ]
    
    results = {}
    
    for test_num, test_name, display_name in tests:
        success = run_test(test_num, test_name)
        results[display_name] = success
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "-"*70)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("-"*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - READY FOR INTEGRATION")
        print("="*70)
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED - FIX BEFORE INTEGRATION")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

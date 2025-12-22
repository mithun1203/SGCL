#!/usr/bin/env python
"""
Quick Demo Script
=================

Run this to quickly verify the SID module is working.

Usage:
    python demo.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("="*60)
    print("SID - Semantic Inconsistency Detector")
    print("Quick Demo")
    print("="*60)
    
    # Import the module
    try:
        from sid import SemanticInconsistencyDetector, create_detector
        print("\n✓ SID module imported successfully")
    except ImportError as e:
        print(f"\n✗ Failed to import SID module: {e}")
        return 1
    
    # Create detector
    try:
        detector = create_detector(backend="rule_based")
        print("✓ Detector created successfully")
    except Exception as e:
        print(f"✗ Failed to create detector: {e}")
        return 1
    
    # Test cases
    test_cases = [
        ("Penguins can fly", True, "Should detect conflict - penguins cannot fly"),
        ("Penguins can swim", False, "No conflict - penguins can swim"),
        ("Birds can fly", False, "No conflict - birds generally can fly"),
        ("Dogs can bark", False, "No conflict - dogs can bark"),
    ]
    
    print("\n" + "-"*60)
    print("Running test cases:")
    print("-"*60)
    
    all_passed = True
    for text, expected_conflict, description in test_cases:
        result = detector.detect_conflict(text)
        
        passed = result.has_conflict == expected_conflict
        status = "✓ PASS" if passed else "✗ FAIL"
        
        if not passed:
            all_passed = False
        
        print(f"\n{status}: '{text}'")
        print(f"   Expected conflict: {expected_conflict}")
        print(f"   Got conflict: {result.has_conflict}")
        print(f"   ({description})")
        
        if result.has_conflict and result.conflicts:
            print(f"   Conflicting fact: {result.conflicts[0].conflicting_triple.to_natural_language()}")
    
    print("\n" + "-"*60)
    
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    
    # Show system info
    print("\nSystem Info:")
    info = detector.get_system_info()
    print(f"  Version: {info['version']}")
    print(f"  NLP Backend: {info['nlp_backend']['backend']}")
    print(f"  Active Rules: {info['active_conflict_rules']}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

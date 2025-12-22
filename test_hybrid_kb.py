"""
Test Hybrid Knowledge Base
==========================

Tests the HybridKnowledgeBase which combines:
1. JSON KB (curated facts)
2. Numberbatch embeddings (semantic similarity)
3. Built-in capability examples

This tests "ConceptNet Mini" functionality for offline use.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from sid.hybrid_kb import HybridKnowledgeBase, create_hybrid_kb, KnowledgeSource


def test_capability_checks():
    """Test capability inference."""
    print("\n" + "=" * 60)
    print("CAPABILITY CHECKS")
    print("=" * 60)
    
    kb = HybridKnowledgeBase()
    print(f"KB Stats: {kb.stats}")
    print()
    
    tests = [
        # (subject, action, expected_can)
        ("penguin", "fly", False),
        ("penguin", "swim", True),
        ("bird", "fly", True),
        ("fish", "swim", True),
        ("fish", "walk", False),
        ("dog", "bark", True),
        ("cat", "meow", True),
        ("cat", "bark", False),
        ("human", "walk", True),
        ("human", "fly", False),
        ("whale", "swim", True),
        ("whale", "fly", False),
        ("bat", "fly", True),
        ("ostrich", "fly", False),
        ("airplane", "fly", True),
    ]
    
    passed = 0
    failed = 0
    
    for subject, action, expected in tests:
        can_do, conf, source = kb.check_capability(subject, action)
        
        result = "CAN" if can_do else "CANNOT"
        expected_str = "CAN" if expected else "CANNOT"
        status = "PASS" if can_do == expected else "FAIL"
        
        if can_do == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"  [{status}] {subject} {result} {action} "
              f"(expected: {expected_str}, conf: {conf:.2f}, src: {source.value})")
    
    print()
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    return passed, failed


def test_relationship_checks():
    """Test relationship verification."""
    print("\n" + "=" * 60)
    print("RELATIONSHIP CHECKS")
    print("=" * 60)
    
    kb = HybridKnowledgeBase()
    
    tests = [
        # (subject, relation, object, expected)
        ("penguin", "is_a", "bird", True),
        ("dog", "is_a", "animal", True),
        ("fish", "is_a", "animal", True),
        ("fire", "has_property", "hot", True),
        ("ice", "has_property", "cold", True),
        ("water", "is_a", "liquid", True),
    ]
    
    passed = 0
    failed = 0
    
    for subj, rel, obj, expected in tests:
        holds, conf, source = kb.check_relationship(subj, rel, obj)
        
        status = "PASS" if holds == expected else "FAIL"
        result = "TRUE" if holds else "FALSE"
        
        if holds == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"  {status} {subj} {rel} {obj}: {result} "
              f"(conf: {conf:.2f}, src: {source.value})")
    
    print()
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    return passed, failed


def test_sid_integration():
    """Test SID detector with hybrid KB knowledge."""
    print("\n" + "=" * 60)
    print("SID INTEGRATION TEST")
    print("=" * 60)
    
    from sid import create_detector
    
    # Create offline-only detector
    detector = create_detector(offline_only=True)
    
    tests = [
        # (statement, should_conflict)
        ("Penguins can fly", True),
        ("Penguins can swim", False),
        ("Birds can fly", False),
        ("Fish can walk", True),
        ("Dogs can bark", False),
        ("Cats can meow", False),
        ("Fire is cold", True),
        ("Ice is cold", False),
        ("The sun is hot", False),
        ("Humans can fly", True),
        ("Whales can swim", False),
        ("Bats can fly", False),  # Bats CAN fly
    ]
    
    passed = 0
    failed = 0
    
    for statement, expected_conflict in tests:
        result = detector.detect_conflict(statement)
        
        status = "PASS" if result.has_conflict == expected_conflict else "FAIL"
        conflict_str = "CONFLICT" if result.has_conflict else "OK"
        expected_str = "CONFLICT" if expected_conflict else "OK"
        
        if result.has_conflict == expected_conflict:
            passed += 1
        else:
            failed += 1
        
        print(f"  {status} \"{statement}\" -> {conflict_str} (expected: {expected_str})")
        
        if result.conflicts:
            for conflict in result.conflicts[:1]:  # Show first conflict
                print(f"      -> {conflict.conflict_type.value}: {conflict.source_triple.to_natural_language()}")
    
    print()
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    return passed, failed


def test_plural_handling():
    """Test that plural forms are handled correctly."""
    print("\n" + "=" * 60)
    print("PLURAL/SINGULAR HANDLING")
    print("=" * 60)
    
    kb = HybridKnowledgeBase()
    
    tests = [
        # Plural inputs should map to singular KB entries
        ("penguins", "fly", False),
        ("birds", "fly", True),
        ("dogs", "bark", True),
        ("cats", "meow", True),
        ("fishes", "swim", True),  # Note: "fishes" is unusual but should work
        ("humans", "walk", True),
    ]
    
    passed = 0
    failed = 0
    
    for subject, action, expected in tests:
        can_do, conf, source = kb.check_capability(subject, action)
        
        status = "PASS" if can_do == expected else "FAIL"
        
        if can_do == expected:
            passed += 1
        else:
            failed += 1
        
        result = "CAN" if can_do else "CANNOT"
        print(f"  {status} {subject} {result} {action} (src: {source.value})")
    
    print()
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    return passed, failed


def test_knowledge_source_priority():
    """Test that JSON KB has priority over inference."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE SOURCE PRIORITY")
    print("=" * 60)
    
    kb = HybridKnowledgeBase()
    
    # These should come from JSON KB (highest priority)
    json_kb_tests = [
        ("penguin", "fly"),
        ("bird", "fly"),
        ("fire", "hot"),
    ]
    
    print("Concepts in JSON KB should use JSON_KB source:")
    for subject, action_or_prop in json_kb_tests:
        # Try capability first
        can_do, conf, source = kb.check_capability(subject, action_or_prop)
        print(f"  {subject}/{action_or_prop}: source={source.value}, conf={conf:.2f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("HYBRID KNOWLEDGE BASE TEST SUITE")
    print("=" * 60)
    print()
    print("Testing the 'ConceptNet Mini' solution:")
    print("- JSON KB for curated facts")
    print("- Built-in examples for capability inference")
    print("- Numberbatch embeddings (if available)")
    
    total_passed = 0
    total_failed = 0
    
    # Run tests
    p, f = test_capability_checks()
    total_passed += p
    total_failed += f
    
    p, f = test_relationship_checks()
    total_passed += p
    total_failed += f
    
    p, f = test_sid_integration()
    total_passed += p
    total_failed += f
    
    p, f = test_plural_handling()
    total_passed += p
    total_failed += f
    
    test_knowledge_source_priority()
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Success Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")
    
    if total_failed == 0:
        print("\n==> ALL TESTS PASSED!")
    else:
        print(f"\n==> {total_failed} tests failed")
    
    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

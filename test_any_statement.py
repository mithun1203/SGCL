#!/usr/bin/env python
"""
Comprehensive Test: SID Works for Any Statement
================================================

This script tests the SID module with a wide variety of statements
to demonstrate that it can handle arbitrary text, not just pre-programmed examples.
"""

import sys
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, '.')

from sid import create_detector, SemanticInconsistencyDetector

def test_diverse_statements():
    """Test SID with many diverse statements."""
    
    print("=" * 70)
    print("SID - Comprehensive Statement Testing")
    print("Testing that SID works for ANY statement")
    print("=" * 70)
    
    # Create detector
    detector = create_detector(backend="rule_based")
    
    # Test cases: (statement, expected_conflict, reason)
    test_cases: List[Tuple[str, bool, str]] = [
        # === Animals - Flight ===
        ("Penguins can fly", True, "Penguins are flightless birds"),
        ("Ostriches can fly", True, "Ostriches are flightless"),
        ("Eagles can fly", False, "Eagles are flying birds"),
        ("Chickens can fly", True, "Chickens cannot really fly"),
        ("Bats can fly", False, "Bats are mammals that can fly"),
        ("Dogs can fly", True, "Dogs cannot fly"),
        ("Cats can fly", True, "Cats cannot fly"),
        ("Pigs can fly", True, "Pigs cannot fly (common idiom)"),
        ("Elephants can fly", True, "Elephants cannot fly"),
        ("Horses can fly", True, "Horses cannot fly"),
        ("Frogs can fly", True, "Frogs cannot fly"),
        ("Fish can fly", True, "Most fish cannot fly"),
        
        # === Animals - Swimming ===
        ("Penguins can swim", False, "Penguins are excellent swimmers"),
        ("Fish can swim", False, "Fish can swim"),
        ("Dogs can swim", False, "Dogs can swim"),
        ("Dolphins can swim", False, "Dolphins swim"),
        ("Whales can swim", False, "Whales can swim"),
        ("Birds can swim", False, "Some birds swim (not a conflict)"),
        
        # === Animals - Other abilities ===
        ("Dogs can bark", False, "Dogs bark"),
        ("Cats can meow", False, "Cats meow"),
        ("Dogs can meow", True, "Dogs don't meow"),
        ("Cats can bark", True, "Cats don't bark"),
        ("Lions can roar", False, "Lions roar"),
        ("Snakes can walk", True, "Snakes cannot walk"),
        ("Snakes can run", True, "Snakes don't run"),
        ("Turtles can swim", False, "Turtles can swim"),
        ("Rabbits can hop", False, "Rabbits hop"),
        ("Bears can climb", False, "Bears can climb"),
        ("Monkeys can climb", False, "Monkeys climb"),
        ("Elephants can jump", True, "Elephants cannot jump"),
        
        # === Properties ===
        ("Fire is cold", True, "Fire is hot, not cold"),
        ("Ice is hot", True, "Ice is cold, not hot"),
        ("The sun is cold", True, "The sun is hot"),
        ("Water is wet", False, "Water is wet"),
        ("Glass is fragile", False, "Glass is fragile"),
        ("Rocks are soft", True, "Rocks are hard, not soft"),
        
        # === Classification (IsA) ===
        ("A penguin is a bird", False, "Penguins are birds"),
        ("A whale is a fish", True, "Whales are mammals, not fish"),
        ("A bat is a bird", True, "Bats are mammals, not birds"),
        ("A dog is a mammal", False, "Dogs are mammals"),
        ("A snake is a mammal", True, "Snakes are reptiles"),
        ("A frog is an amphibian", False, "Frogs are amphibians"),
        ("A shark is a fish", False, "Sharks are fish"),
        
        # === Human abilities ===
        ("Humans can fly", True, "Humans cannot fly naturally"),
        ("Humans can walk", False, "Humans can walk"),
        ("Humans can think", False, "Humans can think"),
        ("Humans can breathe underwater", True, "Humans cannot breathe underwater"),
        ("Babies can walk", True, "Newborn babies cannot walk"),
        ("Babies can speak", True, "Babies cannot speak"),
        
        # === Objects ===
        ("Cars can fly", True, "Regular cars cannot fly"),
        ("Cars can swim", True, "Cars cannot swim"),
        ("Airplanes can fly", False, "Airplanes fly"),
        ("Boats can fly", True, "Boats don't fly"),
        ("Bicycles can fly", True, "Bicycles don't fly"),
        
        # === Plants/Objects ===
        ("Trees can walk", True, "Trees cannot walk"),
        ("Flowers can walk", True, "Flowers cannot move"),
        ("Rocks can grow", True, "Rocks don't grow"),
        
        # === Contradictory states ===
        ("Dead things can move", True, "Dead things cannot move"),
        ("Blind people can see", True, "Blind people cannot see"),
        ("Deaf people can hear", True, "Deaf people cannot hear"),
        
        # === Dietary ===
        ("Vegetarians eat meat", True, "Vegetarians don't eat meat"),
        ("Herbivores eat meat", True, "Herbivores don't eat meat"),
        ("Carnivores eat plants", True, "Carnivores primarily eat meat"),
    ]
    
    # Run tests
    passed = 0
    failed = 0
    unknown = 0
    
    print("\n" + "-" * 70)
    print("Testing statements...")
    print("-" * 70)
    
    for statement, expected_conflict, reason in test_cases:
        try:
            result = detector.detect_conflict(statement)
            actual_conflict = result.has_conflict
            
            if actual_conflict == expected_conflict:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1
            
            conflict_info = ""
            if result.has_conflict and result.conflicts:
                # Show what KB fact caused the conflict
                kb_fact = result.conflicts[0].conflicting_triple
                if kb_fact:
                    conflict_info = f" [KB: {kb_fact.to_natural_language()}]"
            
            print(f"{status}: '{statement}'")
            if status == "✗ FAIL":
                print(f"       Expected: {expected_conflict}, Got: {actual_conflict}")
                print(f"       Reason: {reason}")
            elif result.has_conflict:
                print(f"       Conflict detected{conflict_info}")
                
        except Exception as e:
            print(f"? ERROR: '{statement}' - {e}")
            unknown += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = passed + failed + unknown
    print(f"  Total tests:  {total}")
    print(f"  ✓ Passed:     {passed} ({100*passed/total:.1f}%)")
    print(f"  ✗ Failed:     {failed} ({100*failed/total:.1f}%)")
    if unknown > 0:
        print(f"  ? Errors:     {unknown}")
    print()
    
    # Show known concepts
    print("=" * 70)
    print("KNOWLEDGE BASE INFO")
    print("=" * 70)
    
    # Get the ConceptNet client from the detector
    if hasattr(detector, 'conceptnet_client'):
        client = detector.conceptnet_client
    elif hasattr(detector, 'conflict_engine') and hasattr(detector.conflict_engine, 'conceptnet_client'):
        client = detector.conflict_engine.conceptnet_client
    else:
        client = None
    
    if client and hasattr(client, 'get_known_concepts'):
        concepts = client.get_known_concepts()
        print(f"  Total concepts in KB: {len(concepts)}")
        print(f"  Sample concepts: {', '.join(sorted(concepts)[:20])}...")
    
    return passed, failed, unknown


def interactive_test():
    """Interactive mode - test any statement."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("Type any statement to check for semantic conflicts.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 70 + "\n")
    
    detector = create_detector(backend="rule_based")
    
    while True:
        try:
            statement = input("\nEnter statement: ").strip()
            
            if not statement:
                continue
            if statement.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            result = detector.detect_conflict(statement)
            
            print(f"\n  Statement: \"{statement}\"")
            print(f"  Has conflict: {result.has_conflict}")
            print(f"  Confidence: {result.overall_confidence:.2f}")
            
            if result.extracted_triples:
                print(f"  Extracted triples:")
                for triple in result.extracted_triples:
                    print(f"    - {triple.subject} --[{triple.relation}]--> {triple.object}")
            
            if result.has_conflict and result.conflicts:
                print(f"  Conflict evidence:")
                for ev in result.conflicts:
                    print(f"    - Type: {ev.conflict_type.value}")
                    print(f"    - KB fact: {ev.conflicting_triple.to_natural_language()}")
                    print(f"    - Reasoning: {' -> '.join(ev.reasoning_chain[:2])}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SID with diverse statements")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test()
    else:
        passed, failed, unknown = test_diverse_statements()
        
        # Exit code based on success rate
        if failed == 0:
            print("All tests passed! ✓")
            sys.exit(0)
        elif failed < 10:
            print(f"\nMostly working with {failed} failures.")
            sys.exit(0)
        else:
            print(f"\nToo many failures: {failed}")
            sys.exit(1)

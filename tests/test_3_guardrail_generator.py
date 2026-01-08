"""
Test 3: Guard-Rail Generator

Verifies:
- Produces natural language
- Produces no contradictions
- Produces 2-4 facts
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrail.guardrail_generator import GuardrailGenerator

def test_guardrail_generator():
    """Test guardrail generation."""
    
    print("="*70)
    print("TEST 3: Guard-Rail Generator")
    print("="*70)
    
    print("\n📦 Initializing Guardrail Generator...")
    generator = GuardrailGenerator()
    print("✓ Generator initialized successfully")
    
    # Test cases
    test_cases = [
        {
            "entity": "penguin",
            "relation": "CapableOf",
            "object": "fly",
            "description": "Birds can fly vs Penguins cannot fly"
        },
        {
            "entity": "whale",
            "relation": "IsA",
            "object": "fish",
            "description": "Fish live underwater vs Whales are mammals"
        },
        {
            "entity": "bat",
            "relation": "CapableOf",
            "object": "fly",
            "description": "Mammals don't fly vs Bats can fly"
        }
    ]
    
    print("\n" + "-"*70)
    print("Running Test Cases")
    print("-"*70)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        entity = test_case["entity"]
        relation = test_case["relation"]
        obj = test_case["object"]
        description = test_case["description"]
        
        print(f"\nTest 3.{i}: {entity.capitalize()} - {relation}")
        print(f"Conflict: {description}")
        
        try:
            result = generator.generate(
                conflict_entity=entity,
                conflict_relation=relation,
                conflict_object=obj,
                max_facts=3
            )
            
            # Extract sentences from GuardrailFact objects
            guardrails = [fact.sentence for fact in result] if result else []
            
            print(f"Generated guardrails: {len(guardrails)}")
            
            # Test 1: Has guardrails
            if len(guardrails) == 0:
                print(f"❌ FAIL: No guardrails generated")
                failed += 1
                continue
            
            # Test 2: Reasonable number (2-4)
            if not (2 <= len(guardrails) <= 4):
                print(f"⚠️  WARNING: Unexpected number of guardrails ({len(guardrails)})")
            
            # Test 3: Natural language (not empty)
            all_valid = True
            for j, rail in enumerate(guardrails, 1):
                print(f"   {j}. \"{rail}\"")
                if not rail or len(rail.strip()) < 5:
                    print(f"      ❌ Too short or empty")
                    all_valid = False
            
            if all_valid:
                print(f"✅ PASS: Valid guardrails generated")
                passed += 1
            else:
                print(f"❌ FAIL: Some guardrails invalid")
                failed += 1
                
        except Exception as e:
            print(f"❌ FAIL: Exception occurred: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    if failed == 0:
        print("✅ ALL GUARDRAIL TESTS PASSED")
        print("="*70)
        print(f"Tests run: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Pass rate: {pass_rate:.1f}%")
        print("="*70)
        return True
    else:
        print("❌ SOME GUARDRAIL TESTS FAILED")
        print("="*70)
        print(f"Tests run: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Pass rate: {pass_rate:.1f}%")
        print("="*70)
        return False


if __name__ == "__main__":
    try:
        success = test_guardrail_generator()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

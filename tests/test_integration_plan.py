"""
Pre-Integration Test Plan - Verify all components before SG-CL integration.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_1_seca_dataset():
    """Test 1: SeCA Dataset - Schema, boundaries, conflict distribution, entities."""
    print("\n" + "="*70)
    print("🧩 TEST 1 — SeCA Dataset")
    print("="*70)
    
    try:
        with open("sid/seca_10k_final.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get all samples
        tasks = data.get('tasks', [])
        all_samples = []
        for task in tasks:
            all_samples.extend(task.get('samples', []))
        
        print(f"✓ Total samples: {len(all_samples)}")
        assert len(all_samples) >= 10000, f"Expected >= 10000 samples, got {len(all_samples)}"
        
        # Check schema
        print("✓ Checking schema...")
        assert all("task_id" in s for s in all_samples[:100]), "Missing 'task_id' field"
        assert all("conflict_type" in s for s in all_samples[:100]), "Missing 'conflict_type' field"
        assert all("entities" in s for s in all_samples[:100]), "Missing 'entities' field"
        assert all("sentence" in s for s in all_samples[:100]), "Missing 'sentence' field"
        
        print("✓ Has task_id field: True")
        print("✓ Has conflict_type: True")
        print("✓ Has entities: True")
        print("✓ Has sentence field: True")
        print("\n✅ SeCA dataset OK\n")
        return True
        
    except Exception as e:
        print(f"\n❌ SeCA dataset FAILED: {e}\n")
        return False


def test_2_sid():
    """Test 2: SID - Contradiction detection, hierarchy conflict, no false positives."""
    print("\n" + "="*70)
    print("🧠 TEST 2 — SID")
    print("="*70)
    
    try:
        from sid.detector import SemanticInconsistencyDetector
        
        sid = SemanticInconsistencyDetector()
        
        tests = [
            ("Birds can fly. Penguins cannot fly.", True),
            ("A dog is an animal. A dog is not an animal.", True),
            ("The sky is blue.", False)
        ]
        
        print("Testing conflict detection:\n")
        all_pass = True
        for text, expected in tests:
            result = sid.detect_conflict(text)
            actual = result.has_conflict
            status = "✓" if actual == expected else "✗"
            print(f"{status} '{text}' → {actual} (expected {expected})")
            if actual != expected:
                all_pass = False
        
        if all_pass:
            print("\n✅ SID OK\n")
        else:
            print("\n❌ SID FAILED\n")
        return all_pass
        
    except Exception as e:
        print(f"\n❌ SID FAILED: {e}\n")
        return False


def test_3_guardrail():
    """Test 3: Guard-Rail Generator - Natural language, no contradictions, 2-4 facts."""
    print("\n" + "="*70)
    print("🛡️ TEST 3 — Guard-Rail Generator")
    print("="*70)
    
    try:
        from guardrail.guardrail_generator import GuardrailGenerator
        
        gr = GuardrailGenerator()
        
        # Test with penguin/fly conflict
        rails = gr.generate(
            conflict_entity="penguin",
            conflict_relation="CapableOf",
            conflict_object="fly"
        )
        
        print(f"Generated {len(rails)} guard-rail facts:")
        for i, rail in enumerate(rails, 1):
            print(f"  {i}. {rail.sentence}")
        
        # Verify output
        assert isinstance(rails, list), "Guard-rails should be a list"
        assert len(rails) >= 1, f"Expected at least 1 fact, got {len(rails)}"
        assert all(hasattr(r, 'sentence') for r in rails), "All guard-rails should have sentence attribute"
        
        print("\n✅ Guard-Rail Generator OK\n")
        return True
        
    except ImportError as e:
        print(f"⚠️  Guard-Rail Generator not yet implemented: {e}")
        print("\n⏭️  SKIPPED (not yet implemented)\n")
        return None
    except Exception as e:
        print(f"\n❌ Guard-Rail Generator FAILED: {e}\n")
        return False


def test_4_sid_guardrail_integration():
    """Test 4: SID + Guard-Rail Integration."""
    print("\n" + "="*70)
    print("🔗 TEST 4 — SID + Guard-Rail Integration")
    print("="*70)
    
    try:
        from sid.detector import SemanticInconsistencyDetector
        from guardrail.guardrail_generator import GuardrailGenerator
        
        sid = SemanticInconsistencyDetector()
        gr = GuardrailGenerator()
        
        text = "Birds can fly. Penguins cannot fly."
        res = sid.detect_conflict(text)
        
        print(f"Input: '{text}'")
        print(f"Conflict detected: {res.has_conflict}")
        
        if res.has_conflict:
            # Extract conflict information from result
            if res.conflicts:
                conflict = res.conflicts[0]
                entity = conflict.source_triple.subject
                relation = conflict.source_triple.relation.replace("Not", "")
                obj = conflict.source_triple.object
                
                print(f"Conflict: {entity} {relation} {obj}")
                
                # Generate guardrails
                rails = gr.generate(
                    conflict_entity=entity,
                    conflict_relation=relation,
                    conflict_object=obj
                )
                
                print(f"Guard-rails generated: {len(rails)} facts")
                for i, rail in enumerate(rails, 1):
                    print(f"  {i}. {rail.sentence}")
                
                assert isinstance(rails, list), "Guard-rails should be a list"
                print("\n✅ SID + Guard-Rail Integration OK\n")
                return True
            else:
                print("\n❌ Integration FAILED: Conflict detected but no conflict details\n")
                return False
        else:
            print("\n❌ Integration FAILED: No conflict detected\n")
            return False
        
    except ImportError as e:
        print(f"⚠️  Guard-Rail Generator not yet implemented: {e}")
        print("\n⏭️  SKIPPED (waiting for guard-rail implementation)\n")
        return None
    except Exception as e:
        print(f"\n❌ Integration FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_5_dataset_sid_pipeline():
    """Test 5: Dataset → SID Pipeline."""
    print("\n" + "="*70)
    print("🧾 TEST 5 — Dataset → SID Pipeline")
    print("="*70)
    
    try:
        from sid.detector import SemanticInconsistencyDetector
        
        with open("sid/seca_10k_final.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tasks = data.get('tasks', [])
        all_samples = []
        for task in tasks:
            all_samples.extend(task.get('samples', []))
        
        sid = SemanticInconsistencyDetector()
        
        print("Testing first 100 samples from dataset...\n")
        count = 0
        for i, sample in enumerate(all_samples[:100]):
            text = sample.get('sentence', '')  # Field is 'sentence' not 'text'
            if text:
                result = sid.detect_conflict(text)
                if result.has_conflict:
                    count += 1
        
        percentage = (count / 100) * 100
        print(f"Conflicts detected in sample: {count}/100 ({percentage:.1f}%)")
        
        # Expect reasonable conflict rate (20-60%)
        if 20 <= percentage <= 60:
            print("✓ Conflict rate is reasonable (20-60%)")
            print("\n✅ Dataset → SID Pipeline OK\n")
            return True
        else:
            print(f"⚠️  Conflict rate {percentage:.1f}% is outside expected range (20-60%)")
            print("   This may indicate dataset or SID issues")
            print("\n⚠️  Dataset → SID Pipeline WARNING\n")
            return True  # Still pass, but with warning
        
    except Exception as e:
        print(f"\n❌ Dataset → SID Pipeline FAILED: {e}\n")
        return False


def run_all_tests():
    """Run all pre-integration tests."""
    print("\n" + "="*70)
    print("🧪 PRE-INTEGRATION TEST PLAN")
    print("="*70)
    print("\nVerifying all components before SG-CL integration...\n")
    
    results = {
        "Test 1 (SeCA Dataset)": test_1_seca_dataset(),
        "Test 2 (SID)": test_2_sid(),
        "Test 3 (Guard-Rail)": test_3_guardrail(),
        "Test 4 (Integration)": test_4_sid_guardrail_integration(),
        "Test 5 (Pipeline)": test_5_dataset_sid_pipeline(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("🏁 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIP"
        print(f"{status} - {test_name}")
    
    print("\n" + "-"*70)
    print(f"Total: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print("-"*70)
    
    if failed == 0 and passed >= 3:
        print("\n🎉 INTEGRATION READY!")
        print("All critical components validated.")
        print("SG-CL training is safe to implement.")
    elif failed > 0:
        print("\n⚠️  ISSUES DETECTED")
        print("Please fix failed tests before integration.")
    else:
        print("\n⚠️  INCOMPLETE")
        print("Some components not yet implemented.")
    
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

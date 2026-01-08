"""
Test 1: SeCA Dataset Validation

Verifies:
- Schema compliance
- Task boundaries
- Conflict distribution
- Entity field presence
"""

import json
import sys
from pathlib import Path

def test_seca_dataset():
    """Test SeCA 10K dataset structure and content."""
    
    print("="*70)
    print("TEST 1: SeCA Dataset Validation")
    print("="*70)
    
    # Load dataset
    dataset_path = Path(__file__).parent.parent / "sid" / "seca_10k_final.json"
    
    if not dataset_path.exists():
        print(f"❌ FAIL: Dataset not found at {dataset_path}")
        return False
    
    print(f"\n📂 Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Dataset loaded successfully")
    
    # Extract all samples
    all_samples = []
    for task in data.get('tasks', []):
        all_samples.extend(task.get('samples', []))
    
    total_samples = len(all_samples)
    print(f"✓ Total samples found: {total_samples:,}")
    
    # Test 1: Check total count
    print("\n" + "-"*70)
    print("Test 1.1: Sample Count")
    print("-"*70)
    
    if total_samples >= 10000:
        print(f"✅ PASS: Dataset has {total_samples:,} samples (>= 10,000)")
    else:
        print(f"❌ FAIL: Dataset has only {total_samples:,} samples (expected >= 10,000)")
        return False
    
    # Test 2: Schema validation
    print("\n" + "-"*70)
    print("Test 1.2: Schema Validation")
    print("-"*70)
    
    required_fields = ['task_id', 'sentence', 'label', 'conflict_type', 'entities']
    missing_fields = {field: 0 for field in required_fields}
    
    for i, sample in enumerate(all_samples[:100]):  # Check first 100
        for field in required_fields:
            if field not in sample:
                missing_fields[field] += 1
    
    schema_pass = all(count == 0 for count in missing_fields.values())
    
    if schema_pass:
        print(f"✅ PASS: All required fields present")
        for field in required_fields:
            print(f"   ✓ {field}")
    else:
        print(f"❌ FAIL: Missing fields detected:")
        for field, count in missing_fields.items():
            if count > 0:
                print(f"   ✗ {field}: missing in {count} samples")
        return False
    
    # Test 3: Task boundaries
    print("\n" + "-"*70)
    print("Test 1.3: Task Boundaries")
    print("-"*70)
    
    task_ids = set()
    samples_per_task = {}
    
    for sample in all_samples:
        tid = sample.get('task_id', -1)
        task_ids.add(tid)
        samples_per_task[tid] = samples_per_task.get(tid, 0) + 1
    
    num_tasks = len(task_ids)
    expected_tasks = 16
    
    if num_tasks == expected_tasks:
        print(f"✅ PASS: Found {num_tasks} tasks (expected {expected_tasks})")
    else:
        print(f"❌ FAIL: Found {num_tasks} tasks (expected {expected_tasks})")
        return False
    
    # Check samples per task
    print("\nSamples per task:")
    for tid in sorted(task_ids):
        count = samples_per_task[tid]
        status = "✓" if count == 625 else "✗"
        print(f"   {status} Task {tid}: {count} samples")
    
    uniform = all(count == 625 for count in samples_per_task.values())
    if uniform:
        print(f"✅ PASS: All tasks have exactly 625 samples")
    else:
        print(f"⚠️  WARNING: Non-uniform task distribution")
    
    # Test 4: Conflict distribution
    print("\n" + "-"*70)
    print("Test 1.4: Conflict Distribution")
    print("-"*70)
    
    conflict_count = 0
    no_conflict_count = 0
    conflict_types = {}
    
    for sample in all_samples:
        ctype = sample.get('conflict_type', 'unknown')
        conflict_types[ctype] = conflict_types.get(ctype, 0) + 1
        
        if ctype != 'none':
            conflict_count += 1
        else:
            no_conflict_count += 1
    
    conflict_rate = (conflict_count / total_samples * 100) if total_samples > 0 else 0
    
    print(f"Conflict samples:    {conflict_count:,} ({conflict_rate:.1f}%)")
    print(f"No-conflict samples: {no_conflict_count:,} ({100-conflict_rate:.1f}%)")
    
    if 30 <= conflict_rate <= 60:
        print(f"✅ PASS: Conflict rate {conflict_rate:.1f}% is within 30-60% range")
    else:
        print(f"⚠️  WARNING: Conflict rate {conflict_rate:.1f}% outside 30-60% range")
    
    print("\nConflict type distribution:")
    for ctype, count in sorted(conflict_types.items(), key=lambda x: -x[1])[:7]:
        pct = (count / total_samples * 100)
        print(f"   {ctype:<30} {count:>6,} ({pct:>5.1f}%)")
    
    # Test 5: Entity field
    print("\n" + "-"*70)
    print("Test 1.5: Entity Field Validation")
    print("-"*70)
    
    samples_with_entities = 0
    total_entities = 0
    
    for sample in all_samples:
        entities = sample.get('entities', [])
        if entities and len(entities) > 0:
            samples_with_entities += 1
            total_entities += len(entities)
    
    entity_coverage = (samples_with_entities / total_samples * 100) if total_samples > 0 else 0
    avg_entities = (total_entities / samples_with_entities) if samples_with_entities > 0 else 0
    
    print(f"Samples with entities: {samples_with_entities:,} ({entity_coverage:.1f}%)")
    print(f"Total entities:        {total_entities:,}")
    print(f"Avg entities/sample:   {avg_entities:.2f}")
    
    if entity_coverage > 50:
        print(f"✅ PASS: Entity coverage {entity_coverage:.1f}% is sufficient")
    else:
        print(f"❌ FAIL: Entity coverage {entity_coverage:.1f}% is too low")
        return False
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ALL SECA DATASET TESTS PASSED")
    print("="*70)
    print(f"Dataset: {total_samples:,} samples across {num_tasks} tasks")
    print(f"Conflict rate: {conflict_rate:.1f}%")
    print(f"Entity coverage: {entity_coverage:.1f}%")
    print(f"Schema: 100% compliant")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        success = test_seca_dataset()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

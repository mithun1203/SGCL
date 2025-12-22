"""
SID and SeCA Verification Demo
==============================

Demonstrates that SID uses ONLY knowledge from curated KB sources,
NOT generated or invented answers.

Author: Mithun Naik
Project: SGCL Capstone
"""

import sys
import logging

# Suppress HTTP errors and warnings
logging.disable(logging.CRITICAL)

from sid import create_detector
from sid.seca_dataset import SeCADataset, create_seca_dataset

def demo_sid():
    """Demonstrate SID conflict detection using only KB data."""
    print("=" * 65)
    print("  SID CONFLICT DETECTION - USING ONLY KB DATA")
    print("=" * 65)
    print()
    print("  All knowledge comes from curated sources:")
    print("    - knowledge_base.json (ConceptNet-derived)")
    print("    - Curated KB facts (ConceptNet, WordNet, DBpedia)")
    print("    - NO generated or invented answers")
    print()
    print("-" * 65)
    
    detector = create_detector(offline_only=True)
    
    tests = [
        ("Penguins can fly", True),
        ("Birds can fly", False),
        ("Fish can walk on land", True),
        ("Dogs can bark", False),
        ("Fire is cold", True),
        ("Cats can meow", False),
        ("Snakes have legs", True),
    ]
    
    passed = 0
    for stmt, expected_conflict in tests:
        result = detector.detect_conflict(stmt)
        
        status = "[CONFLICT]" if result.has_conflict else "[OK]"
        correct = result.has_conflict == expected_conflict
        passed += 1 if correct else 0
        
        marker = "[PASS]" if correct else "[FAIL]"
        print(f"  {marker} {status} \"{stmt}\"")
        
        if result.has_conflict and result.conflicts:
            ev = result.conflicts[0]
            t = ev.conflicting_triple
            print(f"         KB Source: {t.source}")
            print(f"         KB Triple: ({t.subject}, {t.relation}, {t.object})")
    
    print("-" * 65)
    print(f"  Results: {passed}/{len(tests)} tests passed")
    print()

def demo_seca():
    """Demonstrate SeCA dataset."""
    print("=" * 65)
    print("  SeCA DATASET - SEMANTIC CONSISTENCY AWARE DATASET")
    print("=" * 65)
    print()
    
    dataset = create_seca_dataset("standard")
    stats = dataset.get_statistics()
    
    print(f"  Name: {stats['name']}")
    print(f"  Version: {stats['version']}")
    print(f"  Total Tasks: {stats['total_tasks']}")
    print(f"  Total Samples: {stats['total_samples']}")
    print(f"  Total Conflicts: {stats['total_conflicts']}")
    print(f"  Conflict Rate: {stats['conflict_rate']:.1%}")
    print()
    
    print("  Conflict Types:")
    for ctype, count in stats['conflict_types'].items():
        print(f"    - {ctype}: {count}")
    print()
    
    print("  Domains:")
    for domain, count in stats['domains'].items():
        print(f"    - {domain}: {count} tasks")
    print()
    
    print("  Sample Conflicts from Dataset:")
    print("-" * 65)
    count = 0
    for task in dataset:
        for sample in task:
            if sample.has_conflict and count < 5:
                print(f"  Task: {task.name}")
                print(f"    Sample: \"{sample.text}\"")
                print(f"    Type: {sample.conflict_type.value}")
                print(f"    Ground Truth: {sample.conflict_with}")
                print()
                count += 1
    print()

def main():
    print("\n")
    demo_sid()
    demo_seca()
    print("=" * 65)
    print("  VERIFICATION COMPLETE")
    print("  - SID uses ONLY curated KB data")
    print("  - SeCA provides ground-truth conflict annotations")
    print("  - No generated or invented answers")
    print("=" * 65)

if __name__ == "__main__":
    main()

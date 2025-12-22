"""
Final Verification - SGCL Capstone Project
==========================================

Verifies that all components are present and working correctly.
"""

from pathlib import Path
import json
import sys


def check_file(path: Path, description: str) -> bool:
    """Check if file exists."""
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}")
    if exists and path.suffix == ".json":
        # Validate JSON
        try:
            with open(path, 'r', encoding='utf-8') as f:
                json.load(f)
            print(f"      (JSON valid)")
        except Exception as e:
            print(f"      (JSON INVALID: {e})")
            return False
    return exists


def verify_project():
    """Verify complete project."""
    
    print("=" * 80)
    print("  SGCL CAPSTONE PROJECT - FINAL VERIFICATION")
    print("=" * 80)
    print()
    
    base_path = Path(__file__).parent
    checks = []
    
    # Core SID Module
    print("1. CORE SID MODULE:")
    checks.append(check_file(base_path / "__init__.py", "Package init"))
    checks.append(check_file(base_path / "detector.py", "Main detector"))
    checks.append(check_file(base_path / "relation_mapper.py", "Relation mapper"))
    checks.append(check_file(base_path / "conflict_engine.py", "Conflict engine"))
    checks.append(check_file(base_path / "hybrid_kb.py", "Hybrid KB"))
    checks.append(check_file(base_path / "knowledge_base.json", "Offline KB"))
    print()
    
    # Tests
    print("2. TEST SUITE:")
    test_dir = base_path.parent / "tests"
    checks.append(check_file(test_dir / "test_sid.py", "SID tests"))
    checks.append(check_file(test_dir / "test_seca_dataset.py", "Dataset tests"))
    checks.append(check_file(base_path.parent / "test_hybrid_kb.py", "KB tests"))
    checks.append(check_file(base_path.parent / "test_any_statement.py", "Integration tests"))
    print()
    
    # Publication Dataset
    print("3. PUBLICATION DATASET:")
    checks.append(check_file(base_path / "seca_publication.py", "Dataset creation script"))
    checks.append(check_file(base_path / "seca_publication_dataset.json", "Full dataset (320 samples)"))
    checks.append(check_file(base_path / "validate_publication.py", "Validation script"))
    checks.append(check_file(base_path / "demo_publication.py", "Demo script"))
    print()
    
    # Evaluation Splits
    print("4. EVALUATION SPLITS:")
    eval_dir = base_path / "evaluation_splits"
    checks.append(check_file(eval_dir / "non_conflict_split.json", "Non-conflict split (240)"))
    checks.append(check_file(eval_dir / "conflict_split.json", "Conflict split (60)"))
    checks.append(check_file(eval_dir / "ambiguous_split.json", "Ambiguous split (20)"))
    checks.append(check_file(eval_dir / "all_split.json", "All samples split (320)"))
    print()
    
    # Documentation
    print("5. DOCUMENTATION:")
    checks.append(check_file(base_path / "SECA_PUBLICATION_GUIDE.md", "Complete guide"))
    checks.append(check_file(base_path / "PUBLICATION_READY.md", "Publication checklist"))
    checks.append(check_file(base_path.parent / "README_COMPLETE.md", "Main README"))
    print()
    
    # Verify dataset content
    print("6. DATASET VALIDATION:")
    dataset_path = base_path / "seca_publication_dataset.json"
    if dataset_path.exists():
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = data['statistics']
        
        checks.append(stats['total_samples'] == 320)
        print(f"  {'✓' if stats['total_samples'] == 320 else '✗'} Total samples: {stats['total_samples']}/320")
        
        checks.append(stats['total_tasks'] == 8)
        print(f"  {'✓' if stats['total_tasks'] == 8 else '✗'} Total tasks: {stats['total_tasks']}/8")
        
        checks.append(len(data['tasks']) == 8)
        print(f"  {'✓' if len(data['tasks']) == 8 else '✗'} Tasks present: {len(data['tasks'])}/8")
        
        all_40 = all(len(t['samples']) == 40 for t in data['tasks'])
        checks.append(all_40)
        print(f"  {'✓' if all_40 else '✗'} All tasks have 40 samples")
        
        label_dist = stats['label_distribution']
        checks.append(label_dist['no_conflict'] >= 100)
        print(f"  {'✓' if label_dist['no_conflict'] >= 100 else '✗'} Non-conflict samples: {label_dist['no_conflict']}")
        
        checks.append(label_dist['conflict'] >= 40)
        print(f"  {'✓' if label_dist['conflict'] >= 40 else '✗'} Conflict samples: {label_dist['conflict']}")
        
        checks.append(label_dist['ambiguous'] >= 20)
        print(f"  {'✓' if label_dist['ambiguous'] >= 20 else '✗'} Ambiguous samples: {label_dist['ambiguous']}")
        
        conflict_types = len(stats.get('conflict_types', {}))
        checks.append(conflict_types >= 4)
        print(f"  {'✓' if conflict_types >= 4 else '✗'} Conflict types: {conflict_types}/4")
    else:
        print("  ✗ Dataset file not found!")
        checks.extend([False] * 8)
    print()
    
    # Summary
    print("=" * 80)
    passed = sum(1 for c in checks if c)
    total = len(checks)
    
    print(f"  VERIFICATION RESULTS: {passed}/{total} checks passed")
    print()
    
    if passed == total:
        print("  ✅ PROJECT IS COMPLETE AND PUBLICATION-READY!")
        print()
        print("  Next Steps:")
        print("    1. Run experiments with baseline models")
        print("    2. Write paper with results")
        print("    3. Submit to conference/journal")
        print("    4. Release dataset publicly")
    else:
        print("  ⚠️  SOME COMPONENTS MISSING")
        print()
        print("  Please ensure all files are generated:")
        print("    python -m sid.seca_publication")
        print("    python -m sid.validate_publication")
    
    print()
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = verify_project()
    sys.exit(0 if success else 1)

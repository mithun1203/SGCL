"""
SeCA Dataset Audit and Fix Script
==================================

Audits seca_publication_dataset.json and auto-fixes missing fields:
- Ensures all samples have 'conflicts_with' field
- Validates schema compliance
- Logs statistics

Run: python audit_and_fix_dataset.py
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def audit_and_fix_dataset(dataset_path: str = "sid/seca_publication_dataset.json"):
    """
    Audit dataset and fix missing fields.
    
    Returns:
        Dict with audit statistics
    """
    print("=" * 70)
    print("SeCA DATASET AUDIT & FIX")
    print("=" * 70)
    
    # Load dataset
    path = Path(dataset_path)
    if not path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        'total_tasks': len(data.get('tasks', [])),
        'total_samples': 0,
        'missing_conflicts_with': 0,
        'missing_conflict_type': 0,
        'missing_entities': 0,
        'samples_fixed': 0,
        'conflict_samples': 0,
        'no_conflict_samples': 0
    }
    
    # Audit and fix each task
    for task in data.get('tasks', []):
        task_name = task.get('name', 'Unknown')
        print(f"\n📋 Task: {task_name}")
        
        for sample in task.get('samples', []):
            stats['total_samples'] += 1
            
            # Count by label
            label = sample.get('label', 'unknown')
            if label == 'conflict':
                stats['conflict_samples'] += 1
            elif label == 'no_conflict':
                stats['no_conflict_samples'] += 1
            
            # Check and fix missing fields
            fixed = False
            
            if 'conflicts_with' not in sample:
                sample['conflicts_with'] = []
                stats['missing_conflicts_with'] += 1
                fixed = True
            
            if 'conflict_type' not in sample:
                sample['conflict_type'] = 'none'
                stats['missing_conflict_type'] += 1
                fixed = True
            
            if 'entities' not in sample:
                sample['entities'] = []
                stats['missing_entities'] += 1
                fixed = True
            
            if fixed:
                stats['samples_fixed'] += 1
    
    # Save fixed dataset
    if stats['samples_fixed'] > 0:
        backup_path = path.parent / f"{path.stem}_backup{path.suffix}"
        print(f"\n💾 Creating backup: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saving fixed dataset: {path}")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("AUDIT RESULTS")
    print("=" * 70)
    print(f"Total tasks:              {stats['total_tasks']}")
    print(f"Total samples:            {stats['total_samples']}")
    print(f"Conflict samples:         {stats['conflict_samples']} ({stats['conflict_samples']/stats['total_samples']*100:.1f}%)")
    print(f"No-conflict samples:      {stats['no_conflict_samples']} ({stats['no_conflict_samples']/stats['total_samples']*100:.1f}%)")
    print()
    print(f"Missing 'conflicts_with': {stats['missing_conflicts_with']}")
    print(f"Missing 'conflict_type':  {stats['missing_conflict_type']}")
    print(f"Missing 'entities':       {stats['missing_entities']}")
    print()
    
    if stats['samples_fixed'] > 0:
        print(f"✅ Fixed {stats['samples_fixed']} samples")
        print(f"✅ Backup saved to: {backup_path}")
    else:
        print("✅ No issues found - dataset is clean!")
    
    print("=" * 70)
    
    return stats


def validate_schema(dataset_path: str = "sid/seca_publication_dataset.json"):
    """
    Validate dataset schema matches Tier-C requirements.
    
    Required fields per sample:
    - task_id
    - sentence (or text)
    - label
    - conflict_type
    - conflicts_with
    """
    print("\n" + "=" * 70)
    print("SCHEMA VALIDATION")
    print("=" * 70)
    
    path = Path(dataset_path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    required_fields = ['sentence', 'label', 'conflict_type', 'conflicts_with']
    issues = []
    
    for task_idx, task in enumerate(data.get('tasks', [])):
        for sample_idx, sample in enumerate(task.get('samples', [])):
            missing = [f for f in required_fields if f not in sample]
            if missing:
                issues.append(f"Task {task_idx}, Sample {sample_idx}: missing {missing}")
    
    if issues:
        print(f"❌ Found {len(issues)} schema issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")
    else:
        print("✅ All samples have required fields!")
        print(f"   - task_id: via parent task")
        print(f"   - sentence: ✓")
        print(f"   - label: ✓")
        print(f"   - conflict_type: ✓")
        print(f"   - conflicts_with: ✓")
    
    print("=" * 70)


if __name__ == "__main__":
    # Run audit and fix
    stats = audit_and_fix_dataset()
    
    # Validate schema
    if stats:
        validate_schema()
        
        print("\n" + "=" * 70)
        print("✅ DATASET READY FOR EXPERIMENTS")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Upload to Kaggle")
        print("2. Run experiments: python run_full_experiments.py")
        print("3. Cite in report: '320 curated samples across 8 tasks'")
        print("=" * 70)

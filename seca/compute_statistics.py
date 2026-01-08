"""
Compute task statistics for SeCA 10K dataset.

Generates statistics table:
- Number of samples per task
- % conflict per task  
- % paraphrase per task (samples with paraphrase variants)
"""

import json
from pathlib import Path
from collections import Counter

def compute_task_statistics(dataset_path: str):
    """Compute and display task statistics."""
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("SECA 10K DATASET STATISTICS")
    print("="*80)
    
    # Overall stats
    total_samples = data.get('total_samples', 0)
    version = data.get('version', 'Unknown')
    print(f"\nDataset: {data.get('name', 'SeCA')}")
    print(f"Version: {version}")
    print(f"Total Samples: {total_samples:,}")
    print(f"Total Tasks: {len(data['tasks'])}")
    
    # Per-task statistics
    print("\n" + "="*80)
    print("PER-TASK STATISTICS")
    print("="*80)
    print(f"{'Task':<25} {'#Samples':<12} {'%Conflict':<12} {'%Paraphrase':<15}")
    print("-" * 80)
    
    overall_conflict = 0
    overall_paraphrase = 0
    
    for task in data['tasks']:
        task_name = task.get('name', 'Unknown')
        samples = task.get('samples', [])
        num_samples = len(samples)
        
        # Count conflicts
        conflict_samples = 0
        paraphrase_samples = 0
        
        for sample in samples:
            # Check if has conflicts
            conflicts_with = sample.get('conflicts_with', [])
            if conflicts_with and len(conflicts_with) > 0:
                conflict_samples += 1
            
            # Check conflict type for paraphrases
            conflict_type = sample.get('conflict_type', '')
            if 'paraphrase' in conflict_type.lower():
                paraphrase_samples += 1
        
        pct_conflict = (conflict_samples / num_samples * 100) if num_samples > 0 else 0
        pct_paraphrase = (paraphrase_samples / num_samples * 100) if num_samples > 0 else 0
        
        overall_conflict += conflict_samples
        overall_paraphrase += paraphrase_samples
        
        print(f"{task_name:<25} {num_samples:<12} {pct_conflict:>6.1f}%{'':<5} {pct_paraphrase:>6.1f}%")
    
    # Overall percentages
    print("-" * 80)
    overall_pct_conflict = (overall_conflict / total_samples * 100) if total_samples > 0 else 0
    overall_pct_paraphrase = (overall_paraphrase / total_samples * 100) if total_samples > 0 else 0
    print(f"{'OVERALL':<25} {total_samples:<12} {overall_pct_conflict:>6.1f}%{'':<5} {overall_paraphrase:>6.1f}%")
    
    # Conflict type distribution
    print("\n" + "="*80)
    print("CONFLICT TYPE DISTRIBUTION")
    print("="*80)
    
    conflict_types = Counter()
    label_distribution = Counter()
    
    for task in data['tasks']:
        for sample in task.get('samples', []):
            conflict_type = sample.get('conflict_type', 'unknown')
            label = sample.get('label', 'unknown')
            conflict_types[conflict_type] += 1
            label_distribution[label] += 1
    
    for ctype, count in conflict_types.most_common():
        pct = (count / total_samples * 100)
        print(f"  {ctype:<30} {count:>6,} ({pct:>5.1f}%)")
    
    print("\n" + "="*80)
    print("LABEL DISTRIBUTION")
    print("="*80)
    
    for label, count in label_distribution.most_common():
        pct = (count / total_samples * 100)
        print(f"  {label:<30} {count:>6,} ({pct:>5.1f}%)")
    
    # Generate markdown table
    print("\n" + "="*80)
    print("MARKDOWN TABLE FOR DOCUMENTATION")
    print("="*80)
    print()
    print("| Task | #Samples | %Conflict | %Paraphrase |")
    print("|------|----------|-----------|-------------|")
    
    for task in data['tasks']:
        task_name = task.get('name', 'Unknown').replace(':', ' -')
        samples = task.get('samples', [])
        num_samples = len(samples)
        
        conflict_samples = sum(1 for s in samples if s.get('conflicts_with'))
        paraphrase_samples = sum(1 for s in samples if 'paraphrase' in s.get('conflict_type', '').lower())
        
        pct_conflict = (conflict_samples / num_samples * 100) if num_samples > 0 else 0
        pct_paraphrase = (paraphrase_samples / num_samples * 100) if num_samples > 0 else 0
        
        print(f"| {task_name} | {num_samples:,} | {pct_conflict:.1f}% | {pct_paraphrase:.1f}% |")
    
    print(f"| **TOTAL** | **{total_samples:,}** | **{overall_pct_conflict:.1f}%** | **{overall_pct_paraphrase:.1f}%** |")
    
    print("\n" + "="*80)
    print("✅ STATISTICS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    # Use the final dataset with entities
    base_dir = Path(__file__).parent.parent
    dataset_file = base_dir / "sid" / "seca_10k_final.json"
    
    if not dataset_file.exists():
        # Fall back to original if final doesn't exist
        dataset_file = base_dir / "sid" / "seca_10k_dataset.json"
    
    compute_task_statistics(str(dataset_file))

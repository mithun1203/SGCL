"""
SeCA Publication Dataset Demo
=============================

Demonstrates the publication dataset with real conflict detection examples.
"""

from pathlib import Path
import json
from sid.seca_publication import SeCAPublicationDataset


def demo_conflict_detection():
    """Demonstrate conflict detection across tasks."""
    
    print("=" * 80)
    print("  SECA PUBLICATION DATASET DEMO")
    print("=" * 80)
    print()
    
    # Load dataset
    dataset_path = Path(__file__).parent / "seca_publication_dataset.json"
    dataset = SeCAPublicationDataset.load(str(dataset_path))
    
    print(f"Dataset: {dataset.name} v{dataset.version}")
    print(f"Total Tasks: {len(dataset.tasks)}")
    print(f"Total Samples: {sum(len(t.samples) for t in dataset.tasks)}")
    print()
    
    # =========================================================================
    # DEMO 1: Base Knowledge (T1)
    # =========================================================================
    print("=" * 80)
    print("  DEMO 1: Base Knowledge (Task 1)")
    print("=" * 80)
    print()
    print("Learning general rules...")
    print()
    
    t1 = dataset.tasks[0]
    for i, sample in enumerate(t1.samples[:3]):
        print(f"  Sample {i+1}: {sample.sentence}")
        print(f"    Label: {sample.label.value}")
    print(f"  ... and {len(t1.samples)-3} more")
    print()
    
    # =========================================================================
    # DEMO 2: Taxonomy (T2)
    # =========================================================================
    print("=" * 80)
    print("  DEMO 2: Taxonomy (Task 2)")
    print("=" * 80)
    print()
    print("Learning hierarchical relations...")
    print()
    
    t2 = dataset.tasks[1]
    for i, sample in enumerate(t2.samples[:3]):
        print(f"  Sample {i+1}: {sample.sentence}")
        print(f"    Label: {sample.label.value}")
    print(f"  ... and {len(t2.samples)-3} more")
    print()
    
    # =========================================================================
    # DEMO 3: Exceptions (T4) - THE KEY TASK
    # =========================================================================
    print("=" * 80)
    print("  DEMO 3: Exceptions (Task 4) - THE KEY LEARNING MOMENT")
    print("=" * 80)
    print()
    print("Learning exceptions to general rules...")
    print("Note: These are NOT conflicts - they are valid exceptions!")
    print()
    
    t4 = dataset.tasks[3]
    exception_samples = [s for s in t4.samples if "cannot" in s.sentence or "do not" in s.sentence][:5]
    
    for i, sample in enumerate(exception_samples):
        print(f"  Exception {i+1}: {sample.sentence}")
        print(f"    Label: {sample.label.value}")
        print(f"    Type: {sample.conflict_type.value}")
    print()
    
    # =========================================================================
    # DEMO 4: Direct Contradictions (T5) - CONFLICT DETECTION
    # =========================================================================
    print("=" * 80)
    print("  DEMO 4: Direct Contradictions (Task 5) - DETECT CONFLICTS!")
    print("=" * 80)
    print()
    print("Now testing if model can detect conflicts with learned knowledge...")
    print()
    
    t5 = dataset.tasks[4]
    conflict_samples = [s for s in t5.samples if s.label.value == "conflict"][:5]
    
    for i, sample in enumerate(conflict_samples):
        print(f"  Conflict {i+1}: {sample.sentence}")
        print(f"    Label: {sample.label.value} ✗")
        print(f"    Conflicts with: {sample.conflicts_with[0]}")
        print(f"    Type: {sample.conflict_type.value}")
        print()
    
    # =========================================================================
    # DEMO 5: Multi-hop Reasoning (T7)
    # =========================================================================
    print("=" * 80)
    print("  DEMO 5: Multi-hop Reasoning (Task 7)")
    print("=" * 80)
    print()
    print("Testing if model can reason across multiple facts...")
    print()
    
    t7 = dataset.tasks[6]
    multihop_conflicts = [s for s in t7.samples if s.label.value == "conflict"][:3]
    
    for i, sample in enumerate(multihop_conflicts):
        print(f"  Example {i+1}: {sample.sentence}")
        print(f"    Label: {sample.label.value} ✗")
        print(f"    Reasoning:")
        for j, step in enumerate(sample.reasoning_chain, 1):
            print(f"      {j}. {step}")
        print()
    
    # =========================================================================
    # DEMO 6: Delayed Contradictions (T8) - FORGETTING TEST
    # =========================================================================
    print("=" * 80)
    print("  DEMO 6: Delayed Contradictions (Task 8) - FORGETTING TEST")
    print("=" * 80)
    print()
    print("After learning 7 tasks, can model still remember Task 1-4 facts?")
    print()
    
    t8 = dataset.tasks[7]
    delayed_samples = [s for s in t8.samples if s.label.value == "ambiguous"][:3]
    
    for i, sample in enumerate(delayed_samples):
        print(f"  Test {i+1}: {sample.sentence}")
        print(f"    Label: {sample.label.value}")
        print(f"    References:")
        for ref in sample.conflicts_with:
            print(f"      - {ref}")
        print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 80)
    print("  SUMMARY: What Makes This Dataset Challenging?")
    print("=" * 80)
    print()
    
    challenges = [
        ("Exception vs Conflict", 
         "Model must distinguish 'Penguins cannot fly' (valid exception) from 'Penguins can fly' (conflict)"),
        
        ("Long-term Memory", 
         "After 280 samples (Tasks 1-7), can model still remember Task 1 facts?"),
        
        ("Multi-hop Reasoning", 
         "Must combine facts: 'Birds fly' + 'Penguins are birds' + 'Penguins cannot fly' → CONFLICT!"),
        
        ("Surface Form Variation", 
         "Same conflict in different forms: statement, question, paraphrase"),
        
        ("Catastrophic Forgetting", 
         "Learning new tasks shouldn't erase old knowledge")
    ]
    
    for i, (challenge, description) in enumerate(challenges, 1):
        print(f"  {i}. {challenge}")
        print(f"     {description}")
        print()
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    print("=" * 80)
    print("  DATASET STATISTICS")
    print("=" * 80)
    print()
    
    stats = dataset.get_statistics()
    
    print(f"  Total Samples: {stats['total_samples']}")
    print(f"  Total Tasks: {stats['total_tasks']}")
    print()
    print("  Label Distribution:")
    for label, count in stats['label_distribution'].items():
        pct = (count / stats['total_samples'] * 100)
        print(f"    {label:15}: {count:3} ({pct:5.1f}%)")
    print()
    print("  Conflict Types:")
    for ctype, count in stats['conflict_types'].items():
        print(f"    {ctype:30}: {count}")
    print()
    
    # =========================================================================
    # KEY EXAMPLES FOR PAPER
    # =========================================================================
    print("=" * 80)
    print("  KEY EXAMPLES FOR PAPER")
    print("=" * 80)
    print()
    
    print("Example 1: Base Rule (T1)")
    print("  'Birds can fly.'")
    print("  → Establishes general knowledge")
    print()
    
    print("Example 2: Taxonomy (T2)")
    print("  'Penguins are birds.'")
    print("  → Establishes hierarchical relation")
    print()
    
    print("Example 3: Exception (T4)")
    print("  'Penguins cannot fly.'")
    print("  → Learns valid exception (NOT a conflict!)")
    print()
    
    print("Example 4: Direct Conflict (T5)")
    print("  'Penguins can fly.'")
    print("  → CONFLICT detected! Contradicts T4")
    print()
    
    print("Example 5: Paraphrase Conflict (T6)")
    print("  'Can penguins fly?'")
    print("  → CONFLICT detected! Same as T5 but as question")
    print()
    
    print("Example 6: Multi-hop Conflict (T7)")
    print("  'Penguins can fly because they are birds.'")
    print("  → CONFLICT! Requires reasoning:")
    print("     1. Birds can fly (T1)")
    print("     2. Penguins are birds (T2)")
    print("     3. ∴ Penguins should fly")
    print("     4. BUT penguins cannot fly (T4)")
    print("     5. → CONTRADICTION!")
    print()
    
    print("Example 7: Delayed Conflict (T8)")
    print("  'Penguins can soar through the sky.'")
    print("  → After 7 tasks, does model remember T4?")
    print()
    
    print("=" * 80)
    print("  DATASET READY FOR PUBLICATION")
    print("=" * 80)
    print()


if __name__ == "__main__":
    demo_conflict_detection()

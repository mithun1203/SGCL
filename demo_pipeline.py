"""
SG-CL Pipeline Integration Demo
===============================

This demonstrates the full SG-CL pipeline integration:

1. SeCA Dataset - Sequential task input
2. SID Analysis - Conflict detection  
3. Gating Controller - Training path decision
4. Guard-Rail Generator - Augmentation for conflicts
5. SG-CL Trainer (stub) - Normal or gated training
6. SCP Evaluator (stub) - Semantic consistency evaluation

Author: Mithun Naik
Project: SGCL Capstone
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def demo_full_pipeline():
    """Demonstrate the complete SG-CL pipeline."""
    print("=" * 70)
    print("  SG-CL PIPELINE INTEGRATION DEMO")
    print("  Symbolic-Gated Continual Learning")
    print("=" * 70)
    print()
    
    from sid import create_detector
    from sid.pipeline import (
        SGCLPipeline, 
        SeCADataset, 
        SCPDataset,
        GatingDecision,
        BasicGuardRailGenerator
    )
    from sid.hybrid_kb import HybridKnowledgeBase
    
    # =========================================================================
    # STEP 1: Create the Pipeline
    # =========================================================================
    print("STEP 1: Creating SG-CL Pipeline")
    print("-" * 50)
    
    detector = create_detector(offline_only=True)
    kb = HybridKnowledgeBase()
    guard_rail_gen = BasicGuardRailGenerator(kb)
    
    pipeline = SGCLPipeline(
        detector=detector,
        guard_rail_generator=guard_rail_gen
    )
    
    print(f"  - Detector: Offline mode (using {kb.stats['json_kb_concepts']} concepts)")
    print(f"  - Guard-Rail Generator: BasicGuardRailGenerator")
    print()
    
    # =========================================================================
    # STEP 2: Create SeCA Dataset (Sequential Tasks)
    # =========================================================================
    print("STEP 2: Creating SeCA Dataset (Sequential Tasks)")
    print("-" * 50)
    
    # Task 1: General bird knowledge
    pipeline.add_task(
        "Birds can fly.",
        "Birds have wings.",
        "Sparrows are birds.",
        "Eagles are birds.",
        name="General Bird Knowledge"
    )
    print("  Task 0: General Bird Knowledge (4 samples)")
    
    # Task 2: Penguin knowledge (contains conflict!)
    pipeline.add_task(
        "Penguins are birds.",
        "Penguins can fly.",  # CONFLICT: Penguins cannot fly!
        "Penguins live in Antarctica.",
        "Penguins can swim very well.",
        name="Penguin Knowledge"
    )
    print("  Task 1: Penguin Knowledge (4 samples) [CONTAINS CONFLICT]")
    
    # Task 3: More bird exceptions
    pipeline.add_task(
        "Ostriches are birds.",
        "Ostriches cannot fly.",  # This is correct
        "Ostriches can run very fast.",
        name="Ostrich Knowledge"
    )
    print("  Task 2: Ostrich Knowledge (3 samples)")
    
    # Task 4: Fish knowledge
    pipeline.add_task(
        "Fish can swim.",
        "Fish live in water.",
        "Whales are mammals.",
        "Whales can swim.",
        name="Aquatic Knowledge"
    )
    print("  Task 3: Aquatic Knowledge (4 samples)")
    
    # Task 5: Another conflict
    pipeline.add_task(
        "Fire is hot.",
        "Ice is cold.",
        "Water is wet.",
        "Fire is cold.",  # CONFLICT!
        name="Property Knowledge"
    )
    print("  Task 4: Property Knowledge (4 samples) [CONTAINS CONFLICT]")
    
    print(f"\n  Total: {len(pipeline.dataset)} tasks, "
          f"{sum(len(t) for t in pipeline.dataset)} samples")
    print()
    
    # =========================================================================
    # STEP 3: Create SCP Probes for Evaluation
    # =========================================================================
    print("STEP 3: Creating SCP Evaluation Probes")
    print("-" * 50)
    
    # Add probes to test semantic consistency
    pipeline.add_scp_probe(
        premise="Penguins are birds",
        hypothesis="Penguins can fly",
        expected=False  # Should be False - penguins can't fly
    )
    
    pipeline.add_scp_probe(
        premise="Birds can fly",
        hypothesis="Sparrows can fly",
        expected=True  # Should be True - sparrows can fly
    )
    
    pipeline.add_scp_probe(
        premise="Ostriches are birds",
        hypothesis="Ostriches can fly", 
        expected=False  # Should be False - ostriches can't fly
    )
    
    pipeline.add_scp_probe(
        premise="Fire is hot",
        hypothesis="Fire is cold",
        expected=False  # Should be False - fire is not cold
    )
    
    print(f"  Added {len(pipeline.scp_dataset)} SCP probes")
    print()
    
    # =========================================================================
    # STEP 4: Run Pipeline Simulation
    # =========================================================================
    print("STEP 4: Running Pipeline Simulation")
    print("-" * 50)
    print()
    
    results = pipeline.run_simulation(verbose=True)
    
    # =========================================================================
    # STEP 5: Show Statistics
    # =========================================================================
    print()
    print("STEP 5: Pipeline Statistics")
    print("-" * 50)
    
    stats = pipeline.get_sid_statistics()
    print(f"  SID Analysis:")
    print(f"    - Total analyzed: {stats['total_analyzed']}")
    print(f"    - Conflicts detected: {stats['conflicts_detected']}")
    print(f"    - Conflict rate: {stats['conflict_rate']:.1%}")
    print(f"    - Gated training rate: {stats['gated_training_rate']:.1%}")
    print(f"    - Avg processing time: {stats['avg_processing_time_ms']:.2f} ms")
    
    print()
    print("  Training Decisions:")
    print(f"    - Normal batches: {results['normal_batches']}")
    print(f"    - Gated batches: {results['gated_batches']}")
    print(f"    - Guard-rails generated: {results['guard_rails_generated']}")
    
    # =========================================================================
    # STEP 6: Show Conflict Details
    # =========================================================================
    print()
    print("STEP 6: Conflict Details")
    print("-" * 50)
    
    for task_detail in results['task_details']:
        if task_detail['conflicts']:
            print(f"\n  Task: {task_detail['task_name']}")
            for conflict in task_detail['conflicts']:
                print(f"    - Conflict: \"{conflict}\"")
            print(f"    - Guard-rails generated:")
            for gr in task_detail['guard_rails']:
                print(f"      * {gr}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("  PIPELINE DEMO COMPLETE")
    print("=" * 70)
    print()
    print("  This demonstrates that SID is fully integrated and ready for:")
    print("    1. Sequential continual learning (SeCA format)")
    print("    2. Automatic conflict detection")
    print("    3. Gating decision (normal vs gated training)")
    print("    4. Guard-rail generation for semantic preservation")
    print("    5. SCP evaluation framework")
    print()
    print("  Next Phase (Phase 2) will implement:")
    print("    - Actual LLM training loop")
    print("    - Advanced guard-rail generation")
    print("    - SCP evaluation with real model")
    print()
    
    return results


def demo_individual_components():
    """Demonstrate individual pipeline components."""
    print()
    print("=" * 70)
    print("  INDIVIDUAL COMPONENT DEMOS")
    print("=" * 70)
    print()
    
    from sid import create_detector
    from sid.pipeline import SIDPipelineAdapter, GatingDecision
    from sid.hybrid_kb import HybridKnowledgeBase
    
    # =========================================================================
    # Demo: SID Pipeline Adapter
    # =========================================================================
    print("1. SID Pipeline Adapter")
    print("-" * 50)
    
    detector = create_detector(offline_only=True)
    adapter = SIDPipelineAdapter(detector)
    
    test_samples = [
        "Penguins can fly",
        "Birds can fly",
        "Fish can walk",
        "Dogs can bark",
    ]
    
    for sample in test_samples:
        result = adapter.analyze(sample)
        decision = "GATED" if result.gating_decision == GatingDecision.GATED_TRAINING else "NORMAL"
        conflict = "CONFLICT" if result.has_conflict else "OK"
        print(f"  \"{sample}\"")
        print(f"    -> {conflict}, Training: {decision}")
        if result.suggested_guard_rails:
            print(f"    -> Guard-rails: {result.suggested_guard_rails[:1]}")
        print()
    
    # =========================================================================
    # Demo: Hybrid Knowledge Base
    # =========================================================================
    print("2. Hybrid Knowledge Base")
    print("-" * 50)
    
    kb = HybridKnowledgeBase()
    
    queries = [
        ("penguin", "fly"),
        ("penguin", "swim"),
        ("bird", "fly"),
        ("ostrich", "fly"),
    ]
    
    for subject, action in queries:
        can_do, conf, source = kb.check_capability(subject, action)
        result = "CAN" if can_do else "CANNOT"
        print(f"  {subject} {result} {action} (conf: {conf:.2f}, src: {source.value})")
    
    print()
    
    # =========================================================================
    # Demo: Guard-Rail Generator
    # =========================================================================
    print("3. Guard-Rail Generator")
    print("-" * 50)
    
    from sid.pipeline import BasicGuardRailGenerator, SIDResult
    
    generator = BasicGuardRailGenerator(kb)
    
    # Create a mock conflict result
    mock_conflict = SIDResult(
        sample="Penguins can fly",
        has_conflict=True,
        conflict_type="direct_contradiction",
        confidence=0.95,
        conflicting_knowledge=["Penguins cannot fly"],
        gating_decision=GatingDecision.GATED_TRAINING
    )
    
    guard_rails = generator.generate([mock_conflict])
    print("  For conflict: \"Penguins can fly\"")
    print("  Generated guard-rails:")
    for gr in guard_rails:
        print(f"    - {gr}")
    
    print()


def main():
    """Run all demos."""
    demo_full_pipeline()
    demo_individual_components()
    
    print("=" * 70)
    print("  ALL DEMOS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

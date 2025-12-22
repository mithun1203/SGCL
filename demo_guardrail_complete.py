"""
Demonstration of Complete SGCL Guardrail System.

Shows the integration of:
1. SID (Semantic Inconsistency Detector)
2. Guardrail Generator (symbolic fact generation)
3. Guardrail Controller (SID-gated batch augmentation)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from guardrail import GuardrailController

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    """Demonstrate complete guardrail system."""
    
    print_section("SGCL Guardrail System Demonstration")
    
    # Initialize controller
    print("\n[1] Initializing Guardrail Controller...")
    controller = GuardrailController(max_guardrails=4)
    print("    ✓ Controller initialized with max_guardrails=4")
    
    # Knowledge base
    knowledge_base = [
        "Birds can fly.",
        "Penguins are birds.",
        "Penguins cannot fly."
    ]
    print(f"\n[2] Knowledge Base ({len(knowledge_base)} facts):")
    for i, fact in enumerate(knowledge_base, 1):
        print(f"    {i}. {fact}")
    
    # Test scenarios
    scenarios = [
        {
            "name": "No Conflict",
            "batch": ["Eagles have sharp talons."],
            "expected_conflict": False
        },
        {
            "name": "Conflict Detected",
            "batch": ["Penguins can fly."],
            "expected_conflict": True
        },
        {
            "name": "Multiple Sentences (Mixed)",
            "batch": ["Birds have wings.", "Penguins can fly."],
            "expected_conflict": True
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print_section(f"Scenario {i}: {scenario['name']}")
        
        batch = scenario['batch']
        print(f"\nInput Batch ({len(batch)} sentences):")
        for j, sentence in enumerate(batch, 1):
            print(f"  {j}. {sentence}")
        
        # Process batch
        result = controller.process_batch(batch, knowledge_base)
        
        # Display results
        print(f"\nConflict Detected: {'YES' if result.has_conflict else 'NO'}")
        
        if result.has_conflict:
            print(f"\nConflict Details:")
            if result.conflict_info:
                print(f"  Entity: {result.conflict_info.get('entity', 'N/A')}")
                print(f"  Relation: {result.conflict_info.get('relation', 'N/A')}")
                print(f"  Object: {result.conflict_info.get('object', 'N/A')}")
            
            print(f"\nGuardrails Generated: {len(result.guardrail_samples)}")
            if result.guardrail_samples:
                print("\nGuardrail Facts:")
                for j, guardrail in enumerate(result.guardrail_samples, 1):
                    print(f"  {j}. {guardrail}")
            
            augmented = result.original_samples + result.guardrail_samples
            print(f"\nAugmented Batch Size: {len(augmented)} (original: {len(batch)}, guardrails: {len(result.guardrail_samples)})")
            print("\nFinal Training Batch:")
            for j, sentence in enumerate(augmented, 1):
                marker = "[ORIGINAL]" if sentence in batch else "[GUARDRAIL]"
                print(f"  {j}. {marker:12} {sentence}")
        else:
            print("\nNo guardrails added (no conflict detected)")
            print(f"Training batch unchanged: {batch}")
    
    # Statistics
    print_section("System Statistics")
    stats = controller.stats
    print(f"\nTotal Batches Processed: {stats['total_batches']}")
    print(f"Conflicts Detected: {stats['conflicts_detected']}")
    print(f"Guardrails Generated: {stats['guardrails_generated']}")
    print(f"Guardrails Injected: {stats['guardrails_injected']}")
    
    if stats['conflicts_detected'] > 0:
        avg_guardrails = stats['guardrails_injected'] / stats['conflicts_detected']
        conflict_rate = (stats['conflicts_detected'] / stats['total_batches']) * 100
        print(f"\nConflict Rate: {conflict_rate:.1f}%")
        print(f"Avg Guardrails per Conflict: {avg_guardrails:.1f}")
    
    # Summary
    print_section("Summary")
    print("\n✅ Guardrail System Features Demonstrated:")
    print("   1. SID-based conflict detection")
    print("   2. Hard gating (guardrails only on conflict)")
    print("   3. Symbolic fact generation (2-4 facts)")
    print("   4. Batch augmentation (original + guardrails)")
    print("   5. Natural language output")
    
    print("\n✅ Guardrail Strategies:")
    print("   - General rule reinforcement (parent class)")
    print("   - Sibling examples (similar entities)")
    print("   - Hierarchy preservation (taxonomic relationships)")
    
    print("\n✅ System Ready for Training Integration")
    print("   Use: result = controller.process_batch(batch, kb)")
    print("   Train on: result.original_samples + result.guardrail_samples")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()

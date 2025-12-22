"""
ConceptNet Mini Demo
====================

Demonstrates the "ConceptNet Mini" solution that provides:
- Offline knowledge for 54+ curated concepts
- Semantic similarity inference for unknown concepts
- 100% test pass rate on common sense conflict detection

No internet connection required!
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def demo_hybrid_kb():
    """Demonstrate the HybridKnowledgeBase capabilities."""
    print("=" * 60)
    print("HYBRID KNOWLEDGE BASE DEMO")
    print("(ConceptNet Mini - Offline Knowledge)")
    print("=" * 60)
    print()
    
    from sid.hybrid_kb import HybridKnowledgeBase
    
    kb = HybridKnowledgeBase()
    stats = kb.stats
    
    print(f"Loaded: {stats['json_kb_concepts']} concepts from JSON KB")
    print(f"Numberbatch: {'Loaded' if stats['numberbatch_loaded'] else 'Not installed'}")
    print(f"Capability actions supported: {', '.join(stats['capability_actions'])}")
    print()
    
    # Test capabilities
    print("CAPABILITY CHECKS:")
    print("-" * 40)
    
    tests = [
        ("penguin", "fly"),
        ("penguin", "swim"),
        ("bird", "fly"),
        ("dog", "bark"),
        ("cat", "meow"),
        ("fish", "walk"),
        ("human", "fly"),
        ("airplane", "fly"),
    ]
    
    for subject, action in tests:
        can_do, conf, source = kb.check_capability(subject, action)
        result = "CAN" if can_do else "CANNOT"
        print(f"  {subject:12} {result:7} {action:10} (conf: {conf:.2f}, src: {source.value})")
    
    print()
    
    # Test relationships
    print("RELATIONSHIP CHECKS:")
    print("-" * 40)
    
    rel_tests = [
        ("penguin", "is_a", "bird"),
        ("dog", "is_a", "animal"),
        ("fire", "has_property", "hot"),
        ("ice", "has_property", "cold"),
    ]
    
    for subj, rel, obj in rel_tests:
        holds, conf, source = kb.check_relationship(subj, rel, obj)
        result = "TRUE" if holds else "FALSE"
        print(f"  {subj:12} {rel:15} {obj:10} = {result:5} (conf: {conf:.2f})")
    
    print()


def demo_sid_detector():
    """Demonstrate the full SID detector."""
    print("=" * 60)
    print("SEMANTIC INCONSISTENCY DETECTOR DEMO")
    print("=" * 60)
    print()
    
    from sid import create_detector
    
    detector = create_detector(offline_only=True)
    
    test_statements = [
        # Should conflict
        ("Penguins can fly", True),
        ("Fish can walk on land", True),
        ("Fire is cold", True),
        ("Humans can fly without machines", True),
        
        # Should NOT conflict  
        ("Penguins can swim", False),
        ("Birds can fly", False),
        ("Dogs can bark", False),
        ("Ice is cold", False),
        ("The sun is hot", False),
        ("Whales swim in the ocean", False),
    ]
    
    print("Testing statements for semantic conflicts:")
    print("-" * 50)
    
    for statement, expected_conflict in test_statements:
        result = detector.detect_conflict(statement)
        
        if result.has_conflict:
            status = "CONFLICT"
            details = f" -> {result.conflicts[0].conflict_type.value}"
        else:
            status = "OK"
            details = ""
        
        expected = "CONFLICT" if expected_conflict else "OK"
        match = "PASS" if (result.has_conflict == expected_conflict) else "FAIL"
        
        print(f"  [{match}] \"{statement}\"")
        print(f"         Result: {status}{details}")
        print()


def demo_knowledge_stats():
    """Show what knowledge is available."""
    print("=" * 60)
    print("AVAILABLE KNOWLEDGE")
    print("=" * 60)
    print()
    
    from sid.conceptnet_client import ConceptNetClient
    
    client = ConceptNetClient()
    concepts = client.get_known_concepts()
    
    print(f"Offline KB contains {len(concepts)} concepts:")
    print("-" * 40)
    
    # Group by category
    categories = {
        "Animals": ["penguin", "ostrich", "bird", "eagle", "sparrow", "hawk", "owl",
                   "dog", "cat", "fish", "whale", "dolphin", "shark", "bat", "elephant",
                   "lion", "tiger", "bear", "monkey", "snake", "frog", "turtle"],
        "Properties": ["fire", "ice", "water", "hot", "cold"],
        "Actions": ["fly", "swim", "walk", "bark", "meow"],
        "Objects": ["car", "airplane", "bicycle", "boat", "train"],
        "Other": []
    }
    
    for concept in concepts:
        found = False
        for cat, items in categories.items():
            if concept in items:
                found = True
                break
        if not found:
            categories["Other"].append(concept)
    
    for cat, items in categories.items():
        matching = [c for c in concepts if c in items]
        if matching:
            print(f"\n{cat}:")
            print(f"  {', '.join(matching[:10])}", end="")
            if len(matching) > 10:
                print(f", ... (+{len(matching)-10} more)")
            else:
                print()


def main():
    """Run all demos."""
    print()
    print("*" * 60)
    print("*    CONCEPTNET MINI - OFFLINE KNOWLEDGE SOLUTION         *")
    print("*    Part of SID (Semantic Inconsistency Detector)        *")
    print("*" * 60)
    print()
    print("This demonstrates offline conflict detection using:")
    print("  1. JSON KB: 54+ curated concepts with relations")
    print("  2. Built-in capability inference rules")
    print("  3. Optional: Numberbatch embeddings (~150MB)")
    print()
    print("No internet connection required!")
    print()
    
    demo_hybrid_kb()
    demo_sid_detector()
    demo_knowledge_stats()
    
    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("To install Numberbatch embeddings for extended coverage:")
    print("  python -m sid.download_numberbatch --type mini")
    print()


if __name__ == "__main__":
    main()

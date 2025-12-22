"""
Guardrail Controller - SID-Gated Training-Time Augmentation
===========================================================

Hard-gated by SID: Only activates when conflict is detected.
Implements the core guardrail logic for SG-CL.

Data Flow:
    Incoming batch → SID analysis → Conflict?
        ├─ NO  → Train normally
        └─ YES → Generate guardrails → Augment batch → Train
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add sid to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sid.detector import SemanticInconsistencyDetector
from guardrail.guardrail_generator import GuardrailGenerator, GuardrailFact


@dataclass
class TrainingBatch:
    """Represents a training batch with optional guardrail augmentation."""
    original_samples: List[str]
    guardrail_samples: List[str]
    has_conflict: bool
    conflict_info: Optional[Dict[str, Any]] = None
    
    @property
    def all_samples(self) -> List[str]:
        """Return combined batch (original + guardrails)."""
        return self.original_samples + self.guardrail_samples
    
    @property
    def size(self) -> int:
        """Total batch size."""
        return len(self.all_samples)


class GuardrailController:
    """
    SID-gated guardrail controller for semantic consistency.
    
    Core Logic:
        IF SID.detect_conflict(batch) == TRUE:
            Activate Guardrail
        ELSE:
            No Guardrail
    
    Paper Description:
        "When a semantic conflict is detected, SG-CL injects a small set of 
        symbolically grounded guardrail facts into the training batch, 
        reinforcing relevant general rules and hierarchies to prevent 
        semantic drift without blocking learning."
    """
    
    def __init__(
        self,
        sid: Optional[SemanticInconsistencyDetector] = None,
        generator: Optional[GuardrailGenerator] = None,
        max_guardrails: int = 4,
        enable_guardrails: bool = True
    ):
        """
        Initialize guardrail controller.
        
        Args:
            sid: Semantic Inconsistency Detector (creates default if None)
            generator: Guardrail generator (creates default if None)
            max_guardrails: Maximum guardrail facts per conflict (2-4 recommended)
            enable_guardrails: Enable/disable guardrails (for ablation studies)
        """
        self.sid = sid or SemanticInconsistencyDetector()
        self.generator = generator or GuardrailGenerator()
        self.max_guardrails = max_guardrails
        self.enable_guardrails = enable_guardrails
        
        # Statistics
        self.stats = {
            'total_batches': 0,
            'conflicts_detected': 0,
            'guardrails_generated': 0,
            'guardrails_injected': 0
        }
    
    def process_batch(
        self,
        batch: List[str],
        knowledge_base: List[str]
    ) -> TrainingBatch:
        """
        Process a training batch with SID-gated guardrail augmentation.
        
        Args:
            batch: List of training sentences
            knowledge_base: Existing knowledge (for conflict detection)
        
        Returns:
            TrainingBatch with optional guardrail augmentation
        
        Example:
            >>> controller = GuardrailController()
            >>> batch = ["Penguins can fly."]
            >>> kb = ["Birds can fly.", "Penguins cannot fly."]
            >>> result = controller.process_batch(batch, kb)
            >>> print(result.has_conflict)  # True
            >>> print(result.guardrail_samples)
            ['Birds can fly.', 'Robins can fly.', 'Sparrows can fly.', 'Penguins are birds.']
        """
        self.stats['total_batches'] += 1
        
        # Step 1: SID Analysis (Hard Gating)
        conflicts = self._detect_conflicts(batch, knowledge_base)
        
        if not conflicts or not self.enable_guardrails:
            # No conflict or guardrails disabled → Train normally
            return TrainingBatch(
                original_samples=batch,
                guardrail_samples=[],
                has_conflict=bool(conflicts)
            )
        
        # Step 2: Conflict detected → Generate Guardrails
        self.stats['conflicts_detected'] += 1
        guardrails = self._generate_guardrails_for_conflicts(conflicts)
        
        # Step 3: Augment batch
        if guardrails:
            self.stats['guardrails_generated'] += len(guardrails)
            self.stats['guardrails_injected'] += len(guardrails)
        
        return TrainingBatch(
            original_samples=batch,
            guardrail_samples=guardrails,
            has_conflict=True,
            conflict_info=conflicts[0] if conflicts else None
        )
    
    def _detect_conflicts(
        self,
        batch: List[str],
        knowledge_base: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Use SID to detect conflicts in batch.
        
        Returns:
            List of conflict dictionaries with entity/relation/object info
        """
        conflicts = []
        
        for sentence in batch:
            # Use SID's detect_conflict method
            result = self.sid.detect_conflict(sentence)
            
            if result.has_conflict and result.conflicts:
                # Extract first conflict evidence
                evidence = result.conflicts[0]
                
                # Get info from source triple (the one being tested)
                source_triple = evidence.source_triple
                
                conflicts.append({
                    'sentence': sentence,
                    'conflicting_fact': evidence.conflicting_triple.to_natural_language(),
                    'conflict_type': evidence.conflict_type.value,
                    'entity': source_triple.subject,
                    'relation': source_triple.relation,
                    'object': source_triple.object
                })
        
        return conflicts
    
    def _generate_guardrails_for_conflicts(
        self,
        conflicts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate guardrail sentences for detected conflicts.
        
        Args:
            conflicts: List of conflict info dicts
        
        Returns:
            List of guardrail sentences (2-4 per conflict)
        """
        all_guardrails = []
        
        for conflict in conflicts[:1]:  # Process first conflict only
            entity = conflict.get('entity')
            relation = conflict.get('relation')
            obj = conflict.get('object')
            
            if not (entity and relation and obj):
                continue
            
            # Normalize relation format for generator (expects ConceptNet format)
            if relation and not relation.startswith('/r/'):
                relation = f'/r/{relation}'
            
            # Generate guardrails using symbolic generator
            facts = self.generator.generate(
                conflict_entity=entity,
                conflict_relation=relation,
                conflict_object=obj,
                max_facts=self.max_guardrails
            )
            
            # Convert to sentences
            guardrail_sentences = [fact.sentence for fact in facts]
            all_guardrails.extend(guardrail_sentences)
        
        return all_guardrails
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get guardrail usage statistics."""
        if self.stats['total_batches'] > 0:
            conflict_rate = self.stats['conflicts_detected'] / self.stats['total_batches']
            avg_guardrails = (
                self.stats['guardrails_injected'] / self.stats['conflicts_detected']
                if self.stats['conflicts_detected'] > 0 else 0
            )
        else:
            conflict_rate = 0
            avg_guardrails = 0
        
        return {
            **self.stats,
            'conflict_rate': conflict_rate,
            'avg_guardrails_per_conflict': avg_guardrails
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            'total_batches': 0,
            'conflicts_detected': 0,
            'guardrails_generated': 0,
            'guardrails_injected': 0
        }


# =============================================================================
# High-Level API
# =============================================================================

def process_training_batch(
    batch: List[str],
    knowledge_base: List[str],
    max_guardrails: int = 4,
    enable_guardrails: bool = True
) -> TrainingBatch:
    """
    Convenience function for batch processing with guardrails.
    
    Args:
        batch: Training sentences
        knowledge_base: Existing knowledge
        max_guardrails: Max guardrail facts (2-4)
        enable_guardrails: Enable/disable (for ablation)
    
    Returns:
        TrainingBatch with augmentation
    
    Example:
        >>> batch = ["Penguins can fly."]
        >>> kb = ["Birds can fly.", "Penguins cannot fly."]
        >>> result = process_training_batch(batch, kb)
        >>> print(f"Conflict: {result.has_conflict}")
        >>> print(f"Guardrails: {result.guardrail_samples}")
    """
    controller = GuardrailController(
        max_guardrails=max_guardrails,
        enable_guardrails=enable_guardrails
    )
    return controller.process_batch(batch, knowledge_base)


if __name__ == "__main__":
    print("=" * 70)
    print("  Guardrail Controller Demo")
    print("=" * 70)
    print()
    
    # Setup
    controller = GuardrailController(max_guardrails=4)
    
    # Scenario 1: No conflict
    print("SCENARIO 1: No Conflict")
    print("-" * 70)
    batch1 = ["Eagles have sharp talons."]
    kb1 = ["Birds can fly.", "Eagles are birds."]
    result1 = controller.process_batch(batch1, kb1)
    
    print(f"Batch: {batch1}")
    print(f"Conflict Detected: {result1.has_conflict}")
    print(f"Guardrails Added: {len(result1.guardrail_samples)}")
    print(f"Final Batch Size: {result1.size}")
    print()
    
    # Scenario 2: Conflict detected
    print("SCENARIO 2: Conflict Detected")
    print("-" * 70)
    batch2 = ["Penguins can fly."]
    kb2 = ["Birds can fly.", "Penguins are birds.", "Penguins cannot fly."]
    result2 = controller.process_batch(batch2, kb2)
    
    print(f"Batch: {batch2}")
    print(f"Knowledge Base: {kb2}")
    print()
    print(f"Conflict Detected: {result2.has_conflict}")
    print(f"Guardrails Added: {len(result2.guardrail_samples)}")
    print()
    print("Guardrail Facts:")
    for i, guardrail in enumerate(result2.guardrail_samples, 1):
        print(f"  {i}. {guardrail}")
    print()
    print(f"Final Batch: {result2.all_samples}")
    print()
    
    # Statistics
    print("=" * 70)
    print("  Statistics")
    print("=" * 70)
    stats = controller.get_statistics()
    print(f"Total Batches: {stats['total_batches']}")
    print(f"Conflicts Detected: {stats['conflicts_detected']}")
    print(f"Conflict Rate: {stats['conflict_rate']:.2%}")
    print(f"Guardrails Injected: {stats['guardrails_injected']}")
    print(f"Avg Guardrails/Conflict: {stats['avg_guardrails_per_conflict']:.1f}")
    print()
    
    print("=" * 70)
    print("  Guardrail System Ready")
    print("=" * 70)

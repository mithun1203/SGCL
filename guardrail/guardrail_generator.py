"""
Guardrail Generator - Symbolic Fact Generation
==============================================

Generates 2-4 symbolically grounded facts when conflict is detected.
Facts are POSITIVE and SUPPORTING - never negations or blocks.

Fact Types:
    A. General rule reinforcement (e.g., "Birds can fly")
    B. Sibling examples (e.g., "Robins can fly", "Sparrows can fly")
    C. Hierarchy preservation (e.g., "Penguins are birds")
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class GuardrailFact:
    """A single guardrail fact to be injected into training batch."""
    sentence: str
    fact_type: str  # 'general_rule', 'sibling_example', 'hierarchy'
    source_relation: str  # e.g., 'CapableOf', 'IsA'
    entities: List[str]
    confidence: float = 1.0


class GuardrailGenerator:
    """
    Generates symbolic guardrail facts for conflict stabilization.
    
    Design Principles:
        1. Small budget: 2-4 facts per conflict
        2. Natural language only (even if source is symbolic)
        3. Positive facts only (no negations)
        4. Grounded in knowledge base (ConceptNet/curated KB)
    """
    
    def __init__(self, kb_path: Optional[str] = None):
        """
        Initialize guardrail generator.
        
        Args:
            kb_path: Path to knowledge base JSON (default: sid/knowledge_base.json)
        """
        if kb_path is None:
            kb_path = Path(__file__).parent.parent / "sid" / "knowledge_base.json"
        
        self.kb_path = Path(kb_path)
        self.kb = self._load_kb()
    
    def _load_kb(self) -> Dict[str, Any]:
        """Load knowledge base."""
        if not self.kb_path.exists():
            return {}
        
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both old and new KB formats
            if 'concepts' in data:
                return data['concepts']
            return data
    
    def generate(
        self,
        conflict_entity: str,
        conflict_relation: str,
        conflict_object: str,
        max_facts: int = 4
    ) -> List[GuardrailFact]:
        """
        Generate guardrail facts for a detected conflict.
        
        Args:
            conflict_entity: Subject of conflict (e.g., "penguin")
            conflict_relation: Relation type (e.g., "CapableOf")
            conflict_object: Object of conflict (e.g., "fly")
            max_facts: Maximum facts to generate (2-4 recommended)
        
        Returns:
            List of guardrail facts (natural language sentences)
        
        Example:
            Conflict: "Penguins can fly" (conflicts with "Penguins cannot fly")
            
            Guardrails generated:
                1. "Birds can fly." (general rule)
                2. "Robins can fly." (sibling example)
                3. "Sparrows can fly." (sibling example)
                4. "Penguins are birds." (hierarchy preservation)
        """
        guardrails = []
        
        # Normalize entity to singular (KB uses singular forms)
        conflict_entity = self._normalize_entity(conflict_entity)
        
        # A. General rule reinforcement
        general_rule = self._generate_general_rule(
            conflict_entity, conflict_relation, conflict_object
        )
        if general_rule:
            guardrails.append(general_rule)
        
        # B. Sibling examples (2 examples)
        siblings = self._generate_sibling_examples(
            conflict_entity, conflict_relation, conflict_object, count=2
        )
        guardrails.extend(siblings)
        
        # C. Hierarchy preservation
        hierarchy = self._generate_hierarchy_fact(conflict_entity)
        if hierarchy:
            guardrails.append(hierarchy)
        
        # Enforce budget: max 2-4 facts
        return guardrails[:max_facts]
    
    def _generate_general_rule(
        self,
        entity: str,
        relation: str,
        obj: str
    ) -> Optional[GuardrailFact]:
        """
        Generate general rule reinforcement.
        
        Example:
            entity="penguin", relation="CapableOf", obj="fly"
            → Find parent class: "bird"
            → Generate: "Birds can fly."
        """
        # Find parent class
        parent_class = self._get_parent_class(entity)
        if not parent_class:
            return None
        
        # Check if parent has this capability
        if not self._has_relation(parent_class, relation, obj):
            return None
        
        # Generate natural language sentence
        sentence = self._relation_to_sentence(parent_class, relation, obj)
        
        return GuardrailFact(
            sentence=sentence,
            fact_type='general_rule',
            source_relation=relation,
            entities=[parent_class, obj],
            confidence=1.0
        )
    
    def _generate_sibling_examples(
        self,
        entity: str,
        relation: str,
        obj: str,
        count: int = 2
    ) -> List[GuardrailFact]:
        """
        Generate sibling examples (same parent class, same capability).
        
        Example:
            entity="penguin", relation="CapableOf", obj="fly"
            → Find siblings: ["robin", "sparrow", "eagle"]
            → Generate: ["Robins can fly.", "Sparrows can fly."]
        """
        siblings = self._get_siblings(entity)
        if not siblings:
            return []
        
        guardrails = []
        for sibling in siblings:
            # Check if sibling has this capability
            if not self._has_relation(sibling, relation, obj):
                continue
            
            sentence = self._relation_to_sentence(sibling, relation, obj)
            guardrails.append(GuardrailFact(
                sentence=sentence,
                fact_type='sibling_example',
                source_relation=relation,
                entities=[sibling, obj],
                confidence=0.9
            ))
            
            # Stop once we have enough examples
            if len(guardrails) >= count:
                break
        
        return guardrails
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity to singular form for KB lookup."""
        entity = entity.lower().strip()
        
        # Simple plural removal (works for most cases)
        if entity.endswith('s') and len(entity) > 3:
            singular = entity[:-1]
            # Check if singular form exists in KB
            if singular in self.kb:
                return singular
        
        return entity
    
    def _generate_hierarchy_fact(self, entity: str) -> Optional[GuardrailFact]:
        """
        Generate hierarchy preservation fact.
        
        Example:
            entity="penguin"
            → "Penguins are birds."
        """
        parent_class = self._get_parent_class(entity)
        if not parent_class:
            return None
        
        # Generate "X is a Y" sentence
        sentence = f"{entity.capitalize()}s are {parent_class}s."
        
        return GuardrailFact(
            sentence=sentence,
            fact_type='hierarchy',
            source_relation='IsA',
            entities=[entity, parent_class],
            confidence=1.0
        )
    
    # =========================================================================
    # Knowledge Base Access Methods
    # =========================================================================
    
    def _get_parent_class(self, entity: str) -> Optional[str]:
        """Get parent class from knowledge base."""
        entity_lower = entity.lower()
        
        if entity_lower in self.kb:
            concept_data = self.kb[entity_lower]
            
            # Handle ConceptNet format (list of relations)
            if isinstance(concept_data, list):
                for rel in concept_data:
                    if '/r/IsA' in rel.get('rel', ''):
                        target = rel.get('end', '')
                        # Extract entity name from ConceptNet URI
                        if '/c/en/' in target:
                            return target.split('/c/en/')[-1]
            
            # Handle old format (dict with relations key)
            elif isinstance(concept_data, dict):
                for rel in concept_data.get('relations', []):
                    if rel['type'] == 'IsA':
                        return rel['target']
        
        return None
    
    def _get_siblings(self, entity: str) -> List[str]:
        """Get sibling entities (same parent class)."""
        parent = self._get_parent_class(entity)
        if not parent:
            return []
        
        siblings = []
        for concept_name, concept_data in self.kb.items():
            if concept_name == entity.lower():
                continue
            
            # Handle ConceptNet format (list)
            if isinstance(concept_data, list):
                for rel in concept_data:
                    if '/r/IsA' in rel.get('rel', ''):
                        target = rel.get('end', '')
                        if parent in target:
                            siblings.append(concept_name)
                            break
            
            # Handle old format (dict)
            elif isinstance(concept_data, dict):
                for rel in concept_data.get('relations', []):
                    if rel['type'] == 'IsA' and rel['target'] == parent:
                        siblings.append(concept_name)
                        break
        
        return siblings[:5]  # Max 5 siblings
    
    def _has_relation(self, entity: str, relation: str, obj: str) -> bool:
        """Check if entity has a relation to object in KB."""
        entity_lower = entity.lower()
        obj_lower = obj.lower()
        
        if entity_lower not in self.kb:
            return False
        
        concept_data = self.kb[entity_lower]
        
        # Normalize relation format
        relation_normalized = relation if relation.startswith('/r/') else f'/r/{relation}'
        
        # Handle ConceptNet format (list)
        if isinstance(concept_data, list):
            for rel in concept_data:
                rel_type = rel.get('rel', '')
                target = rel.get('end', '')
                
                # Match relation type
                if relation_normalized in rel_type and obj_lower in target.lower():
                    return True
        
        # Handle old format (dict)
        elif isinstance(concept_data, dict):
            for rel in concept_data.get('relations', []):
                if rel['type'] == relation and rel['target'].lower() == obj_lower:
                    return True
        
        return False
    
    def _relation_to_sentence(self, entity: str, relation: str, obj: str) -> str:
        """
        Convert symbolic relation to natural language.
        
        Templates based on relation type:
            CapableOf: "{Entity}s can {object}."
            NotCapableOf: "{Entity}s cannot {object}."
            HasProperty: "{Entity}s are {object}."
            HasA: "{Entity}s have {object}."
            NotHasA: "{Entity}s do not have {object}."
        """
        entity_cap = entity.capitalize()
        
        # Strip /r/ prefix if present
        relation_clean = relation.replace('/r/', '') if relation.startswith('/r/') else relation
        
        templates = {
            'CapableOf': f"{entity_cap}s can {obj}.",
            'NotCapableOf': f"{entity_cap}s cannot {obj}.",
            'HasProperty': f"{entity_cap}s are {obj}.",
            'HasA': f"{entity_cap}s have {obj}.",
            'NotHasA': f"{entity_cap}s do not have {obj}.",
            'IsA': f"{entity_cap}s are {obj}s.",
            'PartOf': f"{entity_cap}s are part of {obj}.",
            'UsedFor': f"{entity_cap}s are used for {obj}.",
        }
        
        return templates.get(relation_clean, f"{entity_cap}s {relation_clean.lower()} {obj}.")


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_guardrails(
    conflict_entity: str,
    conflict_relation: str,
    conflict_object: str,
    max_facts: int = 4,
    kb_path: Optional[str] = None
) -> List[str]:
    """
    Convenience function to generate guardrail sentences.
    
    Args:
        conflict_entity: Subject of conflict
        conflict_relation: Relation type
        conflict_object: Object of conflict
        max_facts: Maximum facts (2-4 recommended)
        kb_path: Optional KB path
    
    Returns:
        List of natural language sentences
    
    Example:
        >>> guardrails = generate_guardrails("penguin", "CapableOf", "fly")
        >>> print(guardrails)
        ['Birds can fly.', 'Robins can fly.', 'Sparrows can fly.', 'Penguins are birds.']
    """
    generator = GuardrailGenerator(kb_path)
    facts = generator.generate(conflict_entity, conflict_relation, conflict_object, max_facts)
    return [fact.sentence for fact in facts]


if __name__ == "__main__":
    # Demo: Generate guardrails for penguin/fly conflict
    print("Guardrail Generator Demo")
    print("=" * 50)
    print()
    
    # Conflict: "Penguins can fly" (detected by SID)
    print("Conflict Detected: 'Penguins can fly'")
    print("(Conflicts with: 'Penguins cannot fly')")
    print()
    
    # Generate guardrails
    guardrails = generate_guardrails("penguin", "CapableOf", "fly", max_facts=4)
    
    print("Generated Guardrails:")
    for i, sentence in enumerate(guardrails, 1):
        print(f"  {i}. {sentence}")
    print()
    
    print("These facts will be injected into the training batch")
    print("to stabilize the semantic space during gradient updates.")

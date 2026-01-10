"""
Conflict Detection Engine
=========================

Core logic for detecting semantic conflicts between input statements
and existing knowledge in the ConceptNet knowledge graph.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .models import (
    Triple, ConflictResult, ConflictEvidence, ConflictType,
    SemanticRelation, ExtractionResult
)
from .conceptnet_client import ConceptNetClient, ConceptNetConfig
from .relation_mapper import RelationMapper

logger = logging.getLogger(__name__)


@dataclass 
class ConflictRule:
    """
    A rule for detecting specific types of conflicts.
    
    Attributes:
        name: Rule name/identifier
        description: Human-readable description
        source_relations: Relations in the input that trigger this rule
        conflicting_relations: Relations in KB that indicate conflict
        requires_same_subject: Whether subject must match
        requires_same_object: Whether object must match
        enabled: Whether this rule is active
    """
    name: str
    description: str
    source_relations: List[str]
    conflicting_relations: List[str]
    requires_same_subject: bool = True
    requires_same_object: bool = True
    conflict_type: ConflictType = ConflictType.DIRECT_CONTRADICTION
    enabled: bool = True


class ConflictEngine:
    """
    Engine for detecting semantic conflicts between statements.
    
    This is the core reasoning component that:
    1. Takes extracted triples from input
    2. Queries ConceptNet for related knowledge
    3. Applies conflict detection rules
    4. Performs inheritance-based reasoning
    5. Returns detailed conflict evidence
    
    Features:
        - Direct contradiction detection
        - Inheritance-based conflict detection (e.g., Bird->fly vs Penguin->!fly)
        - Property conflict detection
        - Configurable conflict rules
        - Detailed reasoning chains
    
    Example:
        >>> engine = ConflictEngine()
        >>> input_triple = Triple("penguin", "CapableOf", "fly")
        >>> result = engine.check_conflict(input_triple)
        >>> print(result.has_conflict)  # True
    """
    
    # Default conflict detection rules
    DEFAULT_RULES = [
        ConflictRule(
            name="capability_negation",
            description="CapableOf conflicts with NotCapableOf",
            source_relations=["CapableOf"],
            conflicting_relations=["NotCapableOf"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION
        ),
        ConflictRule(
            name="capability_negation_reverse",
            description="NotCapableOf conflicts with CapableOf",
            source_relations=["NotCapableOf"],
            conflicting_relations=["CapableOf"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION
        ),
        ConflictRule(
            name="property_negation",
            description="HasProperty conflicts with NotHasProperty",
            source_relations=["HasProperty"],
            conflicting_relations=["NotHasProperty"],
            conflict_type=ConflictType.PROPERTY_CONFLICT
        ),
        ConflictRule(
            name="property_negation_reverse",
            description="NotHasProperty conflicts with HasProperty",
            source_relations=["NotHasProperty"],
            conflicting_relations=["HasProperty"],
            conflict_type=ConflictType.PROPERTY_CONFLICT
        ),
        ConflictRule(
            name="isa_negation",
            description="IsA conflicts with NotIsA",
            source_relations=["IsA"],
            conflicting_relations=["NotIsA"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION
        ),
        ConflictRule(
            name="isa_negation_reverse",
            description="NotIsA conflicts with IsA",
            source_relations=["NotIsA"],
            conflicting_relations=["IsA"],
            conflict_type=ConflictType.PROPERTY_CONFLICT
        ),
        ConflictRule(
            name="has_a_negation",
            description="HasA conflicts with NotHasA",
            source_relations=["HasA"],
            conflicting_relations=["NotHasA"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION
        ),
        ConflictRule(
            name="has_a_negation_reverse",
            description="NotHasA conflicts with HasA",
            source_relations=["NotHasA"],
            conflicting_relations=["HasA"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION
        ),
        ConflictRule(
            name="similarity_distinction",
            description="SimilarTo conflicts with DistinctFrom",
            source_relations=["SimilarTo"],
            conflicting_relations=["DistinctFrom"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION
        ),
        ConflictRule(
            name="antonym_similarity",
            description="SimilarTo conflicts with Antonym",
            source_relations=["SimilarTo"],
            conflicting_relations=["Antonym"],
            conflict_type=ConflictType.DIRECT_CONTRADICTION
        ),
    ]
    
    @staticmethod
    def _normalize_concept(concept: str) -> str:
        """
        Normalize a concept for comparison (lowercase, singular form).
        Handles common English plurals.
        """
        word = concept.lower().strip()
        
        # Common irregular plurals
        irregulars = {
            "mice": "mouse", "men": "man", "women": "woman",
            "children": "child", "feet": "foot", "teeth": "tooth",
            "geese": "goose", "people": "person", "oxen": "ox",
            "cacti": "cactus", "fungi": "fungus", "nuclei": "nucleus",
            "wolves": "wolf", "knives": "knife", "lives": "life",
            "wives": "wife", "leaves": "leaf", "selves": "self",
            "halves": "half", "calves": "calf", "loaves": "loaf",
            "thieves": "thief", "shelves": "shelf"
        }
        
        if word in irregulars:
            return irregulars[word]
        
        # Regular plural patterns
        if word.endswith("ies") and len(word) > 3:
            return word[:-3] + "y"
        elif word.endswith("ves") and len(word) > 3:
            return word[:-3] + "f"
        elif word.endswith("es"):
            if any(word.endswith(suf) for suf in ["shes", "ches", "xes", "zes", "sses"]):
                return word[:-2]
            elif word.endswith("oes"):
                return word[:-2]
            else:
                return word[:-1] if len(word) > 2 else word
        elif word.endswith("s") and len(word) > 1 and not word.endswith("ss"):
            return word[:-1]
        
        return word
    
    def _concepts_match(self, concept1: str, concept2: str) -> bool:
        """Check if two concepts match (using normalized comparison)."""
        return self._normalize_concept(concept1) == self._normalize_concept(concept2)
    
    def __init__(
        self,
        conceptnet_client: Optional[ConceptNetClient] = None,
        custom_rules: Optional[List[ConflictRule]] = None,
        enable_inheritance_reasoning: bool = True,
        max_inheritance_depth: int = 3,
        min_conflict_confidence: float = 0.5
    ):
        """
        Initialize the conflict engine.
        
        Args:
            conceptnet_client: ConceptNet client instance
            custom_rules: Additional conflict detection rules
            enable_inheritance_reasoning: Enable inheritance-based conflict detection
            max_inheritance_depth: Maximum depth for inheritance chain traversal
            min_conflict_confidence: Minimum confidence to report a conflict
        """
        self.conceptnet_client = conceptnet_client or ConceptNetClient()
        self.enable_inheritance_reasoning = enable_inheritance_reasoning
        self.max_inheritance_depth = max_inheritance_depth
        self.min_conflict_confidence = min_conflict_confidence
        
        # Initialize rules
        self.rules = self.DEFAULT_RULES.copy()
        if custom_rules:
            self.rules.extend(custom_rules)
        
        # Build rule index for fast lookup
        self._rule_index: Dict[str, List[ConflictRule]] = defaultdict(list)
        for rule in self.rules:
            for rel in rule.source_relations:
                self._rule_index[rel].append(rule)
    
    def check_conflict(
        self,
        triple: Triple,
        additional_context: Optional[List[Triple]] = None
    ) -> Tuple[bool, List[ConflictEvidence]]:
        """
        Check if a single triple conflicts with existing knowledge.
        
        Args:
            triple: The triple to check
            additional_context: Additional triples for context
        
        Returns:
            Tuple of (has_conflict, list_of_evidence)
        """
        conflicts = []
        
        # Get relevant rules for this relation
        applicable_rules = self._rule_index.get(triple.relation, [])
        
        # Query knowledge base for related facts
        kb_facts = self._get_relevant_facts(triple)
        
        # Check direct conflicts
        for rule in applicable_rules:
            if not rule.enabled:
                continue
            
            direct_conflicts = self._check_direct_conflict(triple, kb_facts, rule)
            conflicts.extend(direct_conflicts)
        
        # Check inheritance-based conflicts
        if self.enable_inheritance_reasoning:
            inheritance_conflicts = self._check_inheritance_conflict(triple, kb_facts)
            conflicts.extend(inheritance_conflicts)
        
        # Filter by confidence
        conflicts = [c for c in conflicts if c.confidence >= self.min_conflict_confidence]
        
        return len(conflicts) > 0, conflicts
    
    def check_conflicts_batch(
        self,
        triples: List[Triple]
    ) -> Dict[Triple, List[ConflictEvidence]]:
        """
        Check conflicts for multiple triples.
        
        Args:
            triples: List of triples to check
        
        Returns:
            Dictionary mapping each triple to its conflicts
        """
        results = {}
        
        # Pre-fetch knowledge for all concepts
        all_concepts = set()
        for triple in triples:
            all_concepts.add(triple.subject)
            all_concepts.add(triple.object)
        
        # Query KB (this will cache results)
        for concept in all_concepts:
            self.conceptnet_client.get_edges_for_concept(concept)
        
        # Check each triple
        for triple in triples:
            has_conflict, evidence = self.check_conflict(triple)
            results[triple] = evidence
        
        return results
    
    def _get_relevant_facts(self, triple: Triple) -> List[Triple]:
        """Get facts from knowledge base relevant to the input triple."""
        facts = []
        
        # Get facts about the subject
        subject_edges = self.conceptnet_client.get_edges_for_concept(triple.subject)
        for edge in subject_edges:
            facts.append(edge.to_triple())
        
        # Get facts about the object
        object_edges = self.conceptnet_client.get_edges_for_concept(triple.object)
        for edge in object_edges:
            facts.append(edge.to_triple())
        
        return facts
    
    def _check_direct_conflict(
        self,
        source_triple: Triple,
        kb_facts: List[Triple],
        rule: ConflictRule
    ) -> List[ConflictEvidence]:
        """Check for direct conflicts based on a rule."""
        conflicts = []
        
        for kb_fact in kb_facts:
            # Check if KB fact's relation is in the conflicting relations
            if kb_fact.relation not in rule.conflicting_relations:
                continue
            
            # Check subject match (using normalized comparison)
            if rule.requires_same_subject:
                if not self._concepts_match(source_triple.subject, kb_fact.subject):
                    continue
            
            # Check object match (using normalized comparison)
            if rule.requires_same_object:
                if not self._concepts_match(source_triple.object, kb_fact.object):
                    continue
            
            # Found a conflict!
            reasoning = [
                f"Input states: {source_triple.to_natural_language()}",
                f"Knowledge base contains: {kb_fact.to_natural_language()}",
                f"These are contradictory because {rule.description}"
            ]
            
            evidence = ConflictEvidence(
                source_triple=source_triple,
                conflicting_triple=kb_fact,
                conflict_type=rule.conflict_type,
                reasoning_chain=reasoning,
                confidence=min(source_triple.confidence, kb_fact.confidence)
            )
            conflicts.append(evidence)
        
        return conflicts
    
    def _check_inheritance_conflict(
        self,
        source_triple: Triple,
        kb_facts: List[Triple]
    ) -> List[ConflictEvidence]:
        """
        Check for conflicts through inheritance reasoning.
        
        Example: 
        - Input: "Penguins can fly"
        - KB: "Penguins IsA birds", "Penguins NotCapableOf fly"
        - Conflict: Direct statement contradicts known exception
        
        Or:
        - Input: "All birds can fly"
        - KB: "Penguins IsA birds", "Penguins NotCapableOf fly"
        - Conflict: General rule conflicts with known exception
        """
        conflicts = []
        
        # Get the inheritance chain for the subject
        inheritance_chain = self._get_inheritance_chain(source_triple.subject)
        
        # Check if any parent class has conflicting information
        if source_triple.relation == "CapableOf":
            # Check if subject or its parents have NotCapableOf
            conflicts.extend(
                self._check_capability_inheritance(source_triple, kb_facts, inheritance_chain)
            )
        elif source_triple.relation == "NotCapableOf":
            # Check if subject's parents have CapableOf (exception case)
            conflicts.extend(
                self._check_exception_conflict(source_triple, kb_facts, inheritance_chain)
            )
        elif source_triple.relation == "HasProperty":
            # Check property inheritance
            conflicts.extend(
                self._check_property_inheritance(source_triple, kb_facts, inheritance_chain)
            )
        
        return conflicts
    
    def _get_inheritance_chain(
        self,
        concept: str,
        depth: int = 0
    ) -> List[Tuple[str, int]]:
        """
        Get the inheritance chain (IsA hierarchy) for a concept.
        
        Returns:
            List of (parent_concept, depth) tuples
        """
        if depth >= self.max_inheritance_depth:
            return []
        
        chain = []
        
        # Query IsA relations
        isa_edges = self.conceptnet_client.query_relation(concept, "IsA")
        
        for edge in isa_edges:
            parent = edge.to_triple().object
            chain.append((parent, depth + 1))
            
            # Recursively get parent's parents
            chain.extend(self._get_inheritance_chain(parent, depth + 1))
        
        return chain
    
    def _check_capability_inheritance(
        self,
        source_triple: Triple,
        kb_facts: List[Triple],
        inheritance_chain: List[Tuple[str, int]]
    ) -> List[ConflictEvidence]:
        """Check capability conflicts through inheritance."""
        conflicts = []
        
        # Direct check: Does subject have NotCapableOf for this action?
        for fact in kb_facts:
            if (fact.relation == "NotCapableOf" and 
                self._concepts_match(fact.subject, source_triple.subject) and
                self._concepts_match(fact.object, source_triple.object)):
                
                reasoning = [
                    f"Input claims: {source_triple.to_natural_language()}",
                    f"But knowledge base states: {fact.to_natural_language()}",
                    "This is a direct contradiction of known capability."
                ]
                
                conflicts.append(ConflictEvidence(
                    source_triple=source_triple,
                    conflicting_triple=fact,
                    conflict_type=ConflictType.DIRECT_CONTRADICTION,
                    reasoning_chain=reasoning,
                    confidence=0.95
                ))
        
        return conflicts
    
    def _check_exception_conflict(
        self,
        source_triple: Triple,
        kb_facts: List[Triple],
        inheritance_chain: List[Tuple[str, int]]
    ) -> List[ConflictEvidence]:
        """
        Check if a NotCapableOf statement conflicts with inherited capability.
        
        This detects when something is claimed to NOT do something, 
        but its parent class CAN do it (exception case).
        """
        conflicts = []
        
        # For each parent in inheritance chain
        for parent, depth in inheritance_chain:
            # Check if parent has CapableOf for this action
            parent_capabilities = self.conceptnet_client.query_relation(parent, "CapableOf")
            
            for edge in parent_capabilities:
                capability = edge.to_triple().object
                
                if self._concepts_match(capability, source_triple.object):
                    # This is an exception - the subject cannot do what its parent can
                    reasoning = [
                        f"Input states: {source_triple.to_natural_language()}",
                        f"{source_triple.subject} IsA {parent} (through inheritance)",
                        f"Parent {parent} CapableOf {capability}",
                        f"This means {source_triple.subject} is an exception to the general rule."
                    ]
                    
                    parent_triple = edge.to_triple()
                    
                    # This is actually not always a "conflict" - it might be a valid exception
                    # We flag it as IMPLICIT_CONFLICT for review
                    conflicts.append(ConflictEvidence(
                        source_triple=source_triple,
                        conflicting_triple=parent_triple,
                        conflict_type=ConflictType.INHERITANCE_CONFLICT,
                        reasoning_chain=reasoning,
                        confidence=0.7 - (depth * 0.1),  # Lower confidence for deeper inheritance
                        supporting_facts=[
                            Triple(source_triple.subject, "IsA", parent, source="inheritance")
                        ]
                    ))
        
        return conflicts
    
    def _check_property_inheritance(
        self,
        source_triple: Triple,
        kb_facts: List[Triple],
        inheritance_chain: List[Tuple[str, int]]
    ) -> List[ConflictEvidence]:
        """Check property conflicts through inheritance."""
        conflicts = []
        
        # Direct check: Does subject have NotHasProperty for this property?
        for fact in kb_facts:
            if (fact.relation == "NotHasProperty" and
                self._concepts_match(fact.subject, source_triple.subject) and
                self._concepts_match(fact.object, source_triple.object)):
                
                reasoning = [
                    f"Input claims: {source_triple.to_natural_language()}",
                    f"But knowledge base states: {fact.to_natural_language()}",
                    "This is a direct contradiction of known property."
                ]
                
                conflicts.append(ConflictEvidence(
                    source_triple=source_triple,
                    conflicting_triple=fact,
                    conflict_type=ConflictType.PROPERTY_CONFLICT,
                    reasoning_chain=reasoning,
                    confidence=0.9
                ))
        
        return conflicts
    
    def analyze_statement(
        self,
        text: str,
        relation_mapper: RelationMapper
    ) -> ConflictResult:
        """
        Analyze a natural language statement for conflicts.
        
        This is a high-level method that combines extraction and conflict detection.
        
        Args:
            text: Natural language input
            relation_mapper: Mapper for extracting triples from text
        
        Returns:
            Complete ConflictResult with all findings
        """
        import time
        start_time = time.time()
        
        # Extract triples from text
        extracted_triples = relation_mapper.map_to_triples(text)
        
        if not extracted_triples:
            return ConflictResult(
                has_conflict=False,
                input_text=text,
                extracted_triples=[],
                conflicts=[],
                warnings=["No semantic triples could be extracted from the input"],
                processing_time=time.time() - start_time
            )
        
        # Collect all concepts for KB lookup
        queried_concepts = set()
        for triple in extracted_triples:
            queried_concepts.add(triple.subject)
            queried_concepts.add(triple.object)
        
        # Get all KB facts
        all_kb_facts = []
        for concept in queried_concepts:
            edges = self.conceptnet_client.get_edges_for_concept(concept)
            for edge in edges:
                all_kb_facts.append(edge.to_triple())
        
        # Check conflicts for each extracted triple
        all_conflicts = []
        
        # First, check for conflicts BETWEEN extracted triples (intra-input conflicts)
        for i, triple1 in enumerate(extracted_triples):
            for j, triple2 in enumerate(extracted_triples):
                if i >= j:  # Skip self-comparison and duplicates
                    continue
                
                # Check if these two triples conflict with each other
                inter_conflicts = self._check_inter_triple_conflicts(triple1, triple2)
                all_conflicts.extend(inter_conflicts)
        
        # Check for inheritance-based conflicts (e.g., "birds fly" + "penguins can't fly" where penguin IsA bird)
        inheritance_conflicts = self._check_inheritance_conflicts(extracted_triples)
        all_conflicts.extend(inheritance_conflicts)
        
        # Check for KB-based inheritance conflicts (e.g., extracted "penguins cannot fly" vs KB "birds can fly")
        kb_inheritance_conflicts = self._check_kb_inheritance_conflicts(extracted_triples, all_kb_facts)
        all_conflicts.extend(kb_inheritance_conflicts)
        
        # Then, check conflicts against KB for each extracted triple
        for triple in extracted_triples:
            has_conflict, evidence = self.check_conflict(triple)
            all_conflicts.extend(evidence)
        
        # Build result
        result = ConflictResult(
            has_conflict=len(all_conflicts) > 0,
            input_text=text,
            extracted_triples=extracted_triples,
            conflicts=all_conflicts,
            queried_concepts=list(queried_concepts),
            knowledge_base_facts=all_kb_facts,
            processing_time=time.time() - start_time
        )
        
        return result
    
    def _check_inter_triple_conflicts(
        self,
        triple1: Triple,
        triple2: Triple
    ) -> List[ConflictEvidence]:
        """
        Check if two extracted triples conflict with each other.
        
        For example:
        - "dog IsA animal" conflicts with "dog NotIsA animal"
        - "bird CapableOf fly" conflicts with "bird NotCapableOf fly"
        """
        conflicts = []
        
        # Normalize concepts for comparison
        subject1 = self._normalize_concept(triple1.subject)
        subject2 = self._normalize_concept(triple2.subject)
        object1 = self._normalize_concept(triple1.object)
        object2 = self._normalize_concept(triple2.object)
        
        # Check if subjects and objects match
        if subject1 != subject2 or object1 != object2:
            return conflicts
        
        # Check for opposing relations
        opposing_pairs = [
            ("IsA", "NotIsA"),
            ("CapableOf", "NotCapableOf"),
            ("HasProperty", "NotHasProperty"),
            ("HasA", "NotHasA"),
        ]
        
        for pos_rel, neg_rel in opposing_pairs:
            if (triple1.relation == pos_rel and triple2.relation == neg_rel) or \
               (triple1.relation == neg_rel and triple2.relation == pos_rel):
                
                reasoning = [
                    f"Statement 1: {triple1.to_natural_language()}",
                    f"Statement 2: {triple2.to_natural_language()}",
                    f"These are directly contradictory - same subject and object but opposite relations."
                ]
                
                conflicts.append(ConflictEvidence(
                    source_triple=triple1,
                    conflicting_triple=triple2,
                    conflict_type=ConflictType.DIRECT_CONTRADICTION,
                    reasoning_chain=reasoning,
                    confidence=min(triple1.confidence, triple2.confidence)
                ))
                break
        
        return conflicts
    
    def _check_inheritance_conflicts(
        self,
        triples: List[Triple]
    ) -> List[ConflictEvidence]:
        """
        Check for conflicts based on inheritance relationships.
        
        Example:
        - "birds CapableOf fly" + "penguins NotCapableOf fly" where "penguin IsA bird"
        - "All birds have feathers" + "Penguins are birds" + "Penguins cannot fly"
        
        This checks if:
        1. Triple A says "X has property P"
        2. Triple B says "Y has NOT property P" (or opposite)
        3. KB says "Y IsA X" (inheritance relationship)
        
        Args:
            triples: All extracted triples from the input
        
        Returns:
            List of conflict evidence
        """
        conflicts = []
        
        # Check all pairs of triples
        for i, triple1 in enumerate(triples):
            for j, triple2 in enumerate(triples):
                if i >= j:
                    continue
                
                # Check if triple1 and triple2 have opposing relations and same object
                opposing_pairs = [
                    ("CapableOf", "NotCapableOf"),
                    ("HasProperty", "NotHasProperty"),
                    ("HasA", "NotHasA"),
                ]
                
                for pos_rel, neg_rel in opposing_pairs:
                    # Check if relations are opposing
                    if not ((triple1.relation == pos_rel and triple2.relation == neg_rel) or
                            (triple1.relation == neg_rel and triple2.relation == pos_rel)):
                        continue
                    
                    # Check if objects match
                    obj1 = self._normalize_concept(triple1.object)
                    obj2 = self._normalize_concept(triple2.object)
                    if obj1 != obj2:
                        continue
                    
                    # Check if subject2 IsA subject1 (inheritance)
                    subj1 = self._normalize_concept(triple1.subject)
                    subj2 = self._normalize_concept(triple2.subject)
                    
                    if subj1 == subj2:
                        continue  # Same subject, handled by inter-triple check
                    
                    # Check both directions: subj2 IsA subj1 OR subj1 IsA subj2
                    is_related = False
                    parent = None
                    child = None
                    
                    # Check if subj2 IsA subj1
                    isa_edges = self.conceptnet_client.query_relation(subj2, "IsA", subj1)
                    if isa_edges:
                        is_related = True
                        parent = subj1
                        child = subj2
                    
                    # Check if subj1 IsA subj2
                    if not is_related:
                        isa_edges = self.conceptnet_client.query_relation(subj1, "IsA", subj2)
                        if isa_edges:
                            is_related = True
                            parent = subj2
                            child = subj1
                    
                    if is_related:
                        # Found inheritance-based conflict!
                        parent_triple = triple1 if parent == subj1 else triple2
                        child_triple = triple2 if child == subj2 else triple1
                        
                        reasoning = [
                            f"Parent class statement: {parent_triple.to_natural_language()}",
                            f"Child class statement: {child_triple.to_natural_language()}",
                            f"Inheritance: {child} IsA {parent} (from knowledge base)",
                            f"Exception/Conflict: The child class contradicts the parent class property."
                        ]
                        
                        conflicts.append(ConflictEvidence(
                            source_triple=child_triple,
                            conflicting_triple=parent_triple,
                            conflict_type=ConflictType.INHERITANCE_CONFLICT,
                            reasoning_chain=reasoning,
                            confidence=min(triple1.confidence, triple2.confidence) * 0.9  # Slightly lower confidence
                        ))
        
        return conflicts
    
    def _check_kb_inheritance_conflicts(
        self,
        extracted_triples: List[Triple],
        kb_facts: List[Triple]
    ) -> List[ConflictEvidence]:
        """
        Check for conflicts between extracted triples and KB facts via inheritance.
        
        Example:
        - Extracted: "penguins NotCapableOf fly"
        - KB: "birds CapableOf fly" + "penguin IsA bird"
        - Conflict: Penguin inherits from bird, but contradicts bird's capability
        
        Args:
            extracted_triples: Triples extracted from input
            kb_facts: Facts from knowledge base
        
        Returns:
            List of conflict evidence
        """
        conflicts = []
        
        # For each extracted triple
        for ext_triple in extracted_triples:
            # Check if it contradicts any KB fact via inheritance
            ext_subj = self._normalize_concept(ext_triple.subject)
            ext_obj = self._normalize_concept(ext_triple.object)
            
            # Look for opposing relations in KB
            opposing_pairs = [
                ("CapableOf", "NotCapableOf"),
                ("HasProperty", "NotHasProperty"),
                ("HasA", "NotHasA"),
            ]
            
            for pos_rel, neg_rel in opposing_pairs:
                # Check if ext_triple has one relation
                if ext_triple.relation not in [pos_rel, neg_rel]:
                    continue
                
                # Find opposite relation
                opposite_rel = neg_rel if ext_triple.relation == pos_rel else pos_rel
                
                # Look for KB facts with opposite relation and same object
                for kb_fact in kb_facts:
                    if kb_fact.relation != opposite_rel:
                        continue
                    
                    kb_obj = self._normalize_concept(kb_fact.object)
                    if kb_obj != ext_obj:
                        continue
                    
                    # Same object, opposite relation - check if subjects are related via IsA
                    kb_subj = self._normalize_concept(kb_fact.subject)
                    if kb_subj == ext_subj:
                        continue  # Same subject, will be handled elsewhere
                    
                    # Check if ext_subj IsA kb_subj (child IsA parent)
                    isa_edges = self.conceptnet_client.query_relation(ext_subj, "IsA", kb_subj)
                    if isa_edges:
                        # Found KB-based inheritance conflict!
                        reasoning = [
                            f"Extracted statement: {ext_triple.to_natural_language()}",
                            f"KB parent class: {kb_fact.to_natural_language()}",
                            f"Inheritance: {ext_subj} IsA {kb_subj} (from knowledge base)",
                            f"Conflict: Child class contradicts inherited parent property."
                        ]
                        
                        conflicts.append(ConflictEvidence(
                            source_triple=ext_triple,
                            conflicting_triple=kb_fact,
                            conflict_type=ConflictType.INHERITANCE_CONFLICT,
                            reasoning_chain=reasoning,
                            confidence=ext_triple.confidence * 0.85  # Lower confidence for KB-based
                        ))
        
        return conflicts
    
    def get_potential_conflicts(
        self,
        concept: str
    ) -> List[Tuple[Triple, Triple]]:
        """
        Get all potential conflict pairs involving a concept.
        
        Useful for understanding what knowledge exists about a concept
        that could lead to conflicts.
        
        Returns:
            List of (capability_triple, anti_capability_triple) pairs
        """
        potential_conflicts = []
        
        # Get all facts about the concept
        edges = self.conceptnet_client.get_edges_for_concept(concept)
        triples = [edge.to_triple() for edge in edges]
        
        # Group by relation type
        by_relation = defaultdict(list)
        for triple in triples:
            by_relation[triple.relation].append(triple)
        
        # Find opposing pairs
        opposing_pairs = [
            ("CapableOf", "NotCapableOf"),
            ("HasProperty", "NotHasProperty"),
            ("SimilarTo", "DistinctFrom"),
        ]
        
        for pos_rel, neg_rel in opposing_pairs:
            pos_triples = by_relation.get(pos_rel, [])
            neg_triples = by_relation.get(neg_rel, [])
            
            for pos in pos_triples:
                for neg in neg_triples:
                    if self._concepts_match(pos.object, neg.object):
                        potential_conflicts.append((pos, neg))
        
        return potential_conflicts
    
    def explain_knowledge(self, concept: str) -> str:
        """
        Generate a human-readable explanation of knowledge about a concept.
        
        Useful for debugging and understanding what the system knows.
        """
        lines = [f"Knowledge about '{concept}':", "=" * 40]
        
        # Get all facts
        edges = self.conceptnet_client.get_edges_for_concept(concept)
        triples = [edge.to_triple() for edge in edges]
        
        if not triples:
            lines.append("  No knowledge found in the knowledge base.")
            return "\n".join(lines)
        
        # Group by relation
        by_relation = defaultdict(list)
        for triple in triples:
            by_relation[triple.relation].append(triple)
        
        for relation, rel_triples in sorted(by_relation.items()):
            lines.append(f"\n{relation}:")
            for triple in rel_triples:
                if self._concepts_match(triple.subject, concept):
                    lines.append(f"  → {triple.object}")
                else:
                    lines.append(f"  ← {triple.subject}")
        
        # Check for potential conflicts
        potential_conflicts = self.get_potential_conflicts(concept)
        if potential_conflicts:
            lines.append(f"\nPotential conflict pairs ({len(potential_conflicts)}):")
            for pos, neg in potential_conflicts:
                lines.append(f"  ⚠ {pos.relation}:{pos.object} vs {neg.relation}:{neg.object}")
        
        # Get inheritance chain
        inheritance = self._get_inheritance_chain(concept)
        if inheritance:
            lines.append(f"\nInheritance chain:")
            for parent, depth in inheritance:
                indent = "  " * depth
                lines.append(f"{indent}↑ IsA {parent}")
        
        return "\n".join(lines)
    
    def add_rule(self, rule: ConflictRule) -> None:
        """Add a custom conflict detection rule."""
        self.rules.append(rule)
        for rel in rule.source_relations:
            self._rule_index[rel].append(rule)
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a conflict detection rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a conflict detection rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                return True
        return False
    
    def get_active_rules(self) -> List[ConflictRule]:
        """Get all currently active conflict detection rules."""
        return [r for r in self.rules if r.enabled]

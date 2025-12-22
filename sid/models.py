"""
Data Models for Semantic Inconsistency Detector
================================================

This module defines all data structures used throughout the SID system.
Uses dataclasses and Pydantic for type safety and validation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import json


class SemanticRelation(str, Enum):
    """
    ConceptNet relation types used in semantic reasoning.
    
    These relations represent the semantic connections between concepts
    in the ConceptNet knowledge graph.
    """
    # Capability relations
    CAPABLE_OF = "CapableOf"
    NOT_CAPABLE_OF = "NotCapableOf"
    
    # Type/Instance relations
    IS_A = "IsA"
    INSTANCE_OF = "InstanceOf"
    
    # Property relations
    HAS_PROPERTY = "HasProperty"
    NOT_HAS_PROPERTY = "NotHasProperty"
    HAS_A = "HasA"
    PART_OF = "PartOf"
    
    # Spatial relations
    AT_LOCATION = "AtLocation"
    LOCATED_NEAR = "LocatedNear"
    
    # Causal relations
    CAUSES = "Causes"
    CAUSED_BY = "CausedBy"
    HAS_SUBEVENT = "HasSubevent"
    HAS_FIRST_SUBEVENT = "HasFirstSubevent"
    HAS_LAST_SUBEVENT = "HasLastSubevent"
    HAS_PREREQUISITE = "HasPrerequisite"
    
    # Usage relations
    USED_FOR = "UsedFor"
    RECEIVES_ACTION = "ReceivesAction"
    
    # Desire/Goal relations
    DESIRES = "Desires"
    MOTIVATED_BY_GOAL = "MotivatedByGoal"
    
    # Temporal relations
    CREATED_BY = "CreatedBy"
    MADE_OF = "MadeOf"
    
    # Similarity relations
    SIMILAR_TO = "SimilarTo"
    RELATED_TO = "RelatedTo"
    DISTINCT_FROM = "DistinctFrom"
    
    # Antonymy/Opposition
    ANTONYM = "Antonym"
    
    # Context relations
    HAS_CONTEXT = "HasContext"
    
    # Definition
    DEFINED_AS = "DefinedAs"
    SYMBOL_OF = "SymbolOf"
    
    # Manner
    MANNER_OF = "MannerOf"
    
    # Exception (custom for our framework)
    EXCEPTION_TO = "ExceptionTo"
    
    @classmethod
    def get_negation_pairs(cls) -> Dict[str, str]:
        """Returns pairs of relations that are logical negations of each other."""
        return {
            cls.CAPABLE_OF.value: cls.NOT_CAPABLE_OF.value,
            cls.NOT_CAPABLE_OF.value: cls.CAPABLE_OF.value,
            cls.HAS_PROPERTY.value: cls.NOT_HAS_PROPERTY.value,
            cls.NOT_HAS_PROPERTY.value: cls.HAS_PROPERTY.value,
            cls.SIMILAR_TO.value: cls.DISTINCT_FROM.value,
            cls.DISTINCT_FROM.value: cls.SIMILAR_TO.value,
        }
    
    @classmethod
    def get_conflicting_relations(cls) -> List[Tuple[str, str]]:
        """Returns pairs of relations that indicate potential conflicts."""
        return [
            (cls.CAPABLE_OF.value, cls.NOT_CAPABLE_OF.value),
            (cls.HAS_PROPERTY.value, cls.NOT_HAS_PROPERTY.value),
            (cls.IS_A.value, cls.DISTINCT_FROM.value),
            (cls.SIMILAR_TO.value, cls.ANTONYM.value),
        ]


@dataclass
class Triple:
    """
    Represents a semantic triple (subject, relation, object).
    
    This is the fundamental unit of knowledge representation in the system.
    
    Attributes:
        subject: The subject entity (e.g., "penguin")
        relation: The semantic relation (e.g., "CapableOf")
        object: The object entity (e.g., "fly")
        confidence: Confidence score of this triple [0.0, 1.0]
        source: Source of this triple (e.g., "extraction", "conceptnet")
        metadata: Additional information about the triple
    
    Examples:
        >>> triple = Triple("penguin", "CapableOf", "fly")
        >>> print(triple.to_natural_language())
        "A penguin is capable of fly"
    """
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize the triple components."""
        self.subject = self.subject.lower().strip()
        self.object = self.object.lower().strip()
        # Relation keeps original case (ConceptNet format)
        self.relation = self.relation.strip()
        
    def to_conceptnet_uri(self) -> str:
        """Convert triple to ConceptNet URI format."""
        subj = self.subject.replace(" ", "_")
        obj = self.object.replace(" ", "_")
        return f"/a/[/r/{self.relation}/,/c/en/{subj}/,/c/en/{obj}/]"
    
    def to_natural_language(self) -> str:
        """Convert triple to human-readable format."""
        relation_templates = {
            "CapableOf": "A {subject} is capable of {object}",
            "NotCapableOf": "A {subject} is not capable of {object}",
            "IsA": "A {subject} is a {object}",
            "HasProperty": "A {subject} has the property of being {object}",
            "NotHasProperty": "A {subject} does not have the property of being {object}",
            "HasA": "A {subject} has a {object}",
            "PartOf": "A {subject} is part of {object}",
            "UsedFor": "A {subject} is used for {object}",
            "AtLocation": "A {subject} is at/in {object}",
            "Causes": "{subject} causes {object}",
            "HasPrerequisite": "{subject} requires {object}",
            "InstanceOf": "A {subject} is an instance of {object}",
            "RelatedTo": "{subject} is related to {object}",
            "Antonym": "{subject} is the opposite of {object}",
            "SimilarTo": "{subject} is similar to {object}",
            "DistinctFrom": "{subject} is distinct from {object}",
            "Desires": "A {subject} desires {object}",
            "MadeOf": "A {subject} is made of {object}",
            "ReceivesAction": "A {subject} can be {object}",
        }
        
        template = relation_templates.get(
            self.relation, 
            f"{{subject}} [{self.relation}] {{object}}"
        )
        return template.format(subject=self.subject, object=self.object)
    
    def get_negation(self) -> Optional['Triple']:
        """Returns the logical negation of this triple if applicable."""
        negation_pairs = SemanticRelation.get_negation_pairs()
        if self.relation in negation_pairs:
            return Triple(
                subject=self.subject,
                relation=negation_pairs[self.relation],
                object=self.object,
                confidence=self.confidence,
                source=f"negation_of_{self.source}",
                metadata={"original": self.to_dict()}
            )
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Triple':
        """Create Triple from dictionary."""
        return cls(
            subject=data["subject"],
            relation=data["relation"],
            object=data["object"],
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "unknown"),
            metadata=data.get("metadata", {})
        )
    
    def __hash__(self):
        return hash((self.subject, self.relation, self.object))
    
    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.subject == other.subject and 
                self.relation == other.relation and 
                self.object == other.object)


@dataclass
class ConceptNetEdge:
    """
    Represents an edge from ConceptNet knowledge graph.
    
    Attributes:
        start: Start node (subject concept)
        end: End node (object concept)
        relation: Relation type
        weight: Edge weight (importance/frequency)
        surfaceText: Human-readable version of the edge
        sources: List of sources that contributed this edge
    """
    start: str
    end: str
    relation: str
    weight: float
    surface_text: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    dataset: str = "conceptnet"
    license: str = "cc:by-sa/4.0"
    
    def to_triple(self) -> Triple:
        """Convert ConceptNet edge to Triple format."""
        # Extract concept from URI format /c/en/concept_name
        start_concept = self._extract_concept(self.start)
        end_concept = self._extract_concept(self.end)
        relation_type = self._extract_relation(self.relation)
        
        return Triple(
            subject=start_concept,
            relation=relation_type,
            object=end_concept,
            confidence=min(1.0, self.weight / 10.0),  # Normalize weight
            source="conceptnet",
            metadata={
                "surface_text": self.surface_text,
                "original_weight": self.weight,
                "sources": self.sources
            }
        )
    
    @staticmethod
    def _extract_concept(uri: str) -> str:
        """Extract concept name from ConceptNet URI."""
        # /c/en/penguin -> penguin
        if uri.startswith("/c/"):
            parts = uri.split("/")
            if len(parts) >= 4:
                return parts[3].replace("_", " ")
        return uri
    
    @staticmethod
    def _extract_relation(uri: str) -> str:
        """Extract relation name from ConceptNet URI."""
        # /r/CapableOf -> CapableOf
        if uri.startswith("/r/"):
            return uri[3:]
        return uri
    
    @classmethod
    def from_api_response(cls, edge_data: Dict[str, Any]) -> 'ConceptNetEdge':
        """Create ConceptNetEdge from API response or simplified format."""
        # Handle both full API response format and simplified offline KB format
        start = edge_data.get("start", "")
        if isinstance(start, dict):
            start = start.get("@id", "")
        
        end = edge_data.get("end", "")
        if isinstance(end, dict):
            end = end.get("@id", "")
        
        rel = edge_data.get("rel", "")
        if isinstance(rel, dict):
            rel = rel.get("@id", "")
        
        sources = edge_data.get("sources", [])
        if sources and isinstance(sources[0], dict):
            sources = [s.get("@id", "") for s in sources]
        
        return cls(
            start=start,
            end=end,
            relation=rel,
            weight=edge_data.get("weight", 1.0),
            surface_text=edge_data.get("surfaceText"),
            sources=sources if isinstance(sources, list) else [],
            dataset=edge_data.get("dataset", "conceptnet"),
            license=edge_data.get("license", "cc:by-sa/4.0")
        )


@dataclass
class ExtractionResult:
    """
    Result of entity and relation extraction from text.
    
    Attributes:
        original_text: The input text
        entities: List of extracted entities with their types
        triples: List of extracted semantic triples
        extraction_method: Method used for extraction
        processing_time: Time taken for extraction in seconds
    """
    original_text: str
    entities: List[Dict[str, Any]]
    triples: List[Triple]
    extraction_method: str = "spacy"
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def get_subjects(self) -> List[str]:
        """Get all unique subjects from extracted triples."""
        return list(set(t.subject for t in self.triples))
    
    def get_objects(self) -> List[str]:
        """Get all unique objects from extracted triples."""
        return list(set(t.object for t in self.triples))
    
    def get_relations(self) -> List[str]:
        """Get all unique relations from extracted triples."""
        return list(set(t.relation for t in self.triples))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "entities": self.entities,
            "triples": [t.to_dict() for t in self.triples],
            "extraction_method": self.extraction_method,
            "processing_time": self.processing_time,
            "warnings": self.warnings
        }


class ConflictType(str, Enum):
    """Types of semantic conflicts that can be detected."""
    DIRECT_CONTRADICTION = "direct_contradiction"  # A CapableOf X vs A NotCapableOf X
    INHERITANCE_CONFLICT = "inheritance_conflict"  # Penguin IsA Bird, Bird CapableOf Fly, Penguin NotCapableOf Fly
    PROPERTY_CONFLICT = "property_conflict"  # A HasProperty X vs A HasProperty NotX
    TYPE_CONFLICT = "type_conflict"  # A IsA X vs A IsA Y (where X,Y are disjoint)
    IMPLICIT_CONFLICT = "implicit_conflict"  # Inferred conflict through reasoning
    NO_CONFLICT = "no_conflict"


@dataclass
class ConflictEvidence:
    """
    Evidence supporting a detected conflict.
    
    Attributes:
        source_triple: The input triple being evaluated
        conflicting_triple: The triple from knowledge base that conflicts
        conflict_type: Type of conflict detected
        reasoning_chain: Step-by-step reasoning that led to conflict detection
        confidence: Confidence in this conflict detection
    """
    source_triple: Triple
    conflicting_triple: Triple
    conflict_type: ConflictType
    reasoning_chain: List[str]
    confidence: float = 1.0
    supporting_facts: List[Triple] = field(default_factory=list)
    
    def explain(self) -> str:
        """Generate human-readable explanation of the conflict."""
        explanation = [
            f"Conflict Detected ({self.conflict_type.value}):",
            f"  Input: {self.source_triple.to_natural_language()}",
            f"  Conflicts with: {self.conflicting_triple.to_natural_language()}",
            f"  Confidence: {self.confidence:.2%}",
            "  Reasoning:"
        ]
        for i, step in enumerate(self.reasoning_chain, 1):
            explanation.append(f"    {i}. {step}")
        
        if self.supporting_facts:
            explanation.append("  Supporting facts:")
            for fact in self.supporting_facts:
                explanation.append(f"    - {fact.to_natural_language()}")
        
        return "\n".join(explanation)


@dataclass
class ConflictResult:
    """
    Complete result of conflict detection analysis.
    
    This is the main output of the SemanticInconsistencyDetector.
    
    Attributes:
        has_conflict: Whether any conflict was detected
        input_text: Original input text
        extracted_triples: Triples extracted from input
        conflicts: List of detected conflicts with evidence
        queried_concepts: Concepts that were looked up in knowledge base
        total_kb_facts: Total facts retrieved from knowledge base
        processing_time: Total processing time in seconds
    """
    has_conflict: bool
    input_text: str
    extracted_triples: List[Triple]
    conflicts: List[ConflictEvidence]
    queried_concepts: List[str] = field(default_factory=list)
    knowledge_base_facts: List[Triple] = field(default_factory=list)
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    @property
    def conflict_count(self) -> int:
        """Number of conflicts detected."""
        return len(self.conflicts)
    
    @property
    def most_severe_conflict(self) -> Optional[ConflictEvidence]:
        """Returns the conflict with highest confidence."""
        if not self.conflicts:
            return None
        return max(self.conflicts, key=lambda c: c.confidence)
    
    @property
    def conflict_types(self) -> List[ConflictType]:
        """Returns list of all conflict types found."""
        return list(set(c.conflict_type for c in self.conflicts))
    
    def get_conflicting_facts(self) -> List[Triple]:
        """Get all triples that conflict with input."""
        return [c.conflicting_triple for c in self.conflicts]
    
    def summary(self) -> str:
        """Generate a summary of the conflict detection result."""
        if not self.has_conflict:
            return f"No conflicts detected for: '{self.input_text}'"
        
        lines = [
            f"CONFLICT DETECTED for: '{self.input_text}'",
            f"Number of conflicts: {self.conflict_count}",
            f"Conflict types: {[ct.value for ct in self.conflict_types]}",
            "",
            "Details:"
        ]
        for i, conflict in enumerate(self.conflicts, 1):
            lines.append(f"\n[Conflict {i}]")
            lines.append(conflict.explain())
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "has_conflict": self.has_conflict,
            "input_text": self.input_text,
            "extracted_triples": [t.to_dict() for t in self.extracted_triples],
            "conflicts": [
                {
                    "source_triple": c.source_triple.to_dict(),
                    "conflicting_triple": c.conflicting_triple.to_dict(),
                    "conflict_type": c.conflict_type.value,
                    "reasoning_chain": c.reasoning_chain,
                    "confidence": c.confidence,
                    "supporting_facts": [f.to_dict() for f in c.supporting_facts]
                }
                for c in self.conflicts
            ],
            "queried_concepts": self.queried_concepts,
            "knowledge_base_facts": [t.to_dict() for t in self.knowledge_base_facts],
            "processing_time": self.processing_time,
            "warnings": self.warnings
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConflictResult':
        """Create ConflictResult from dictionary."""
        extracted_triples = [Triple.from_dict(t) for t in data.get("extracted_triples", [])]
        kb_facts = [Triple.from_dict(t) for t in data.get("knowledge_base_facts", [])]
        
        conflicts = []
        for c in data.get("conflicts", []):
            conflicts.append(ConflictEvidence(
                source_triple=Triple.from_dict(c["source_triple"]),
                conflicting_triple=Triple.from_dict(c["conflicting_triple"]),
                conflict_type=ConflictType(c["conflict_type"]),
                reasoning_chain=c["reasoning_chain"],
                confidence=c["confidence"],
                supporting_facts=[Triple.from_dict(f) for f in c.get("supporting_facts", [])]
            ))
        
        return cls(
            has_conflict=data["has_conflict"],
            input_text=data["input_text"],
            extracted_triples=extracted_triples,
            conflicts=conflicts,
            queried_concepts=data.get("queried_concepts", []),
            knowledge_base_facts=kb_facts,
            processing_time=data.get("processing_time", 0.0),
            warnings=data.get("warnings", [])
        )


@dataclass
class BatchConflictResult:
    """Result of batch conflict detection on multiple inputs."""
    results: List[ConflictResult]
    total_inputs: int
    inputs_with_conflicts: int
    total_conflicts: int
    total_processing_time: float
    
    @property
    def conflict_rate(self) -> float:
        """Percentage of inputs that had conflicts."""
        if self.total_inputs == 0:
            return 0.0
        return self.inputs_with_conflicts / self.total_inputs
    
    def summary(self) -> str:
        """Generate batch summary."""
        return (
            f"Batch Conflict Detection Results:\n"
            f"  Total inputs: {self.total_inputs}\n"
            f"  Inputs with conflicts: {self.inputs_with_conflicts}\n"
            f"  Conflict rate: {self.conflict_rate:.2%}\n"
            f"  Total conflicts: {self.total_conflicts}\n"
            f"  Processing time: {self.total_processing_time:.2f}s"
        )

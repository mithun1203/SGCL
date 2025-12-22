"""
Relation Mapper Module
======================

Maps natural language expressions to semantic relations (ConceptNet format).
Uses pattern matching, dependency parsing, and verb frame analysis.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .models import Triple, SemanticRelation
from .entity_extractor import EntityExtractor, Entity, EntityType

logger = logging.getLogger(__name__)


@dataclass
class RelationPattern:
    """
    A pattern for matching natural language to semantic relations.
    
    Attributes:
        relation: The target semantic relation
        patterns: List of regex patterns to match
        verb_frames: List of verb lemmas that trigger this relation
        keywords: Keywords that indicate this relation
        requires_negation: Whether negation detection affects this relation
        priority: Higher priority patterns are checked first
    """
    relation: str
    patterns: List[str] = field(default_factory=list)
    verb_frames: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    requires_negation: bool = False
    negated_relation: Optional[str] = None
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation": self.relation,
            "patterns": self.patterns,
            "verb_frames": self.verb_frames,
            "keywords": self.keywords,
            "requires_negation": self.requires_negation,
            "negated_relation": self.negated_relation,
            "priority": self.priority
        }


class RelationMapper:
    """
    Maps natural language text to semantic relation triples.
    
    This is a core component of the SID module that converts natural language
    statements into structured knowledge triples suitable for conflict detection.
    
    Features:
        - Pattern-based relation extraction
        - Verb frame analysis
        - Negation handling
        - Dependency-based extraction (when NLP backend available)
    
    Example:
        >>> mapper = RelationMapper()
        >>> triples = mapper.map_to_triples("Penguins cannot fly")
        >>> for triple in triples:
        ...     print(triple.to_natural_language())
        "A penguin is not capable of fly"
    """
    
    # Default relation patterns
    DEFAULT_PATTERNS = [
        # CapableOf / NotCapableOf patterns
        RelationPattern(
            relation="CapableOf",
            patterns=[
                r"(\w+)\s+can\s+(\w+)",
                r"(\w+)\s+is\s+able\s+to\s+(\w+)",
                r"(\w+)\s+is\s+capable\s+of\s+(\w+ing|\w+)",
                r"(\w+)\s+has\s+the\s+ability\s+to\s+(\w+)",
                r"(\w+)\s+are\s+able\s+to\s+(\w+)",
            ],
            verb_frames=["can", "able", "capable"],
            keywords=["can", "able to", "capable of"],
            requires_negation=True,
            negated_relation="NotCapableOf",
            priority=10
        ),
        
        # NotCapableOf patterns (explicit negation)
        RelationPattern(
            relation="NotCapableOf",
            patterns=[
                r"(\w+)\s+cannot\s+(\w+)",
                r"(\w+)\s+can't\s+(\w+)",
                r"(\w+)\s+can\s+not\s+(\w+)",
                r"(\w+)\s+is\s+not\s+able\s+to\s+(\w+)",
                r"(\w+)\s+is\s+unable\s+to\s+(\w+)",
                r"(\w+)\s+are\s+unable\s+to\s+(\w+)",
                r"(\w+)\s+is\s+incapable\s+of\s+(\w+ing|\w+)",
                r"(\w+)\s+are\s+not\s+able\s+to\s+(\w+)",
            ],
            verb_frames=["cannot", "can't", "unable", "incapable"],
            keywords=["cannot", "can't", "unable to", "incapable of"],
            requires_negation=False,  # Already negative
            priority=11  # Higher priority than CapableOf
        ),
        
        # IsA patterns
        RelationPattern(
            relation="IsA",
            patterns=[
                r"(\w+)\s+is\s+a\s+(?:type\s+of\s+)?(\w+)",
                r"(\w+)\s+is\s+an?\s+(\w+)",
                r"(\w+)\s+are\s+(\w+)s?",
                r"a\s+(\w+)\s+is\s+a\s+(?:kind\s+of\s+)?(\w+)",
                r"(\w+)\s+belongs?\s+to\s+(?:the\s+)?(\w+)",
            ],
            verb_frames=["is", "be", "belong"],
            keywords=["is a", "are", "type of", "kind of"],
            priority=9
        ),
        
        # HasProperty / NotHasProperty patterns
        RelationPattern(
            relation="HasProperty",
            patterns=[
                r"(\w+)\s+is\s+(\w+)",
                r"(\w+)\s+are\s+(\w+)",
                r"(\w+)\s+has\s+the\s+property\s+(?:of\s+)?(\w+)",
                r"(\w+)\s+is\s+characterized\s+by\s+(\w+)",
            ],
            verb_frames=["is", "are"],
            keywords=["is", "are", "property"],
            requires_negation=True,
            negated_relation="NotHasProperty",
            priority=5
        ),
        
        # HasA patterns
        RelationPattern(
            relation="HasA",
            patterns=[
                r"(\w+)\s+has\s+(?:a\s+)?(\w+)",
                r"(\w+)\s+have\s+(?:a\s+)?(\w+)",
                r"(\w+)\s+possesses?\s+(?:a\s+)?(\w+)",
                r"(\w+)\s+owns?\s+(?:a\s+)?(\w+)",
            ],
            verb_frames=["have", "has", "possess", "own"],
            keywords=["has", "have", "possess"],
            requires_negation=True,
            negated_relation="NotHasA",
            priority=7
        ),
        
        # NotHasA patterns (explicit negation)
        RelationPattern(
            relation="NotHasA",
            patterns=[
                r"(\w+)\s+(?:does\s+not|doesn't|do\s+not|don't)\s+have\s+(?:a\s+|any\s+)?(\w+)",
                r"(\w+)\s+(?:has|have)\s+no\s+(\w+)",
                r"(\w+)\s+lacks?\s+(?:a\s+)?(\w+)",
                r"(\w+)\s+is\s+without\s+(?:a\s+)?(\w+)",
            ],
            verb_frames=["lack", "without"],
            keywords=["no", "lack", "without", "doesn't have"],
            requires_negation=False,  # Already negative
            priority=8  # Higher than HasA
        ),
        
        # PartOf patterns
        RelationPattern(
            relation="PartOf",
            patterns=[
                r"(\w+)\s+is\s+(?:a\s+)?part\s+of\s+(\w+)",
                r"(\w+)\s+belongs?\s+to\s+(\w+)",
                r"(\w+)\s+is\s+(?:a\s+)?component\s+of\s+(\w+)",
            ],
            verb_frames=["part", "belong", "component"],
            keywords=["part of", "belongs to", "component"],
            priority=6
        ),
        
        # AtLocation patterns
        RelationPattern(
            relation="AtLocation",
            patterns=[
                r"(\w+)\s+(?:is|are)\s+(?:found\s+)?(?:in|at|on)\s+(?:the\s+)?(\w+)",
                r"(\w+)\s+lives?\s+in\s+(?:the\s+)?(\w+)",
                r"(\w+)\s+(?:is|are)\s+located\s+(?:in|at)\s+(?:the\s+)?(\w+)",
                r"you\s+(?:can\s+)?find\s+(\w+)\s+(?:in|at)\s+(?:the\s+)?(\w+)",
            ],
            verb_frames=["live", "located", "found"],
            keywords=["in", "at", "located", "found in"],
            priority=6
        ),
        
        # UsedFor patterns
        RelationPattern(
            relation="UsedFor",
            patterns=[
                r"(\w+)\s+is\s+used\s+(?:for|to)\s+(\w+)",
                r"(\w+)\s+(?:is|are)\s+for\s+(\w+)",
                r"you\s+use\s+(\w+)\s+(?:for|to)\s+(\w+)",
                r"(\w+)\s+helps?\s+(?:to\s+)?(\w+)",
            ],
            verb_frames=["use", "used", "help"],
            keywords=["used for", "for", "helps"],
            priority=6
        ),
        
        # Causes patterns
        RelationPattern(
            relation="Causes",
            patterns=[
                r"(\w+)\s+causes?\s+(\w+)",
                r"(\w+)\s+leads?\s+to\s+(\w+)",
                r"(\w+)\s+results?\s+in\s+(\w+)",
                r"(\w+)\s+makes?\s+(\w+)",
            ],
            verb_frames=["cause", "lead", "result", "make"],
            keywords=["causes", "leads to", "results in"],
            priority=7
        ),
        
        # HasPrerequisite patterns
        RelationPattern(
            relation="HasPrerequisite",
            patterns=[
                r"(\w+)\s+requires?\s+(\w+)",
                r"(\w+)\s+needs?\s+(\w+)",
                r"to\s+(\w+)\s+you\s+need\s+(\w+)",
                r"(\w+)\s+depends?\s+on\s+(\w+)",
            ],
            verb_frames=["require", "need", "depend"],
            keywords=["requires", "needs", "depends on"],
            priority=6
        ),
        
        # Desires patterns
        RelationPattern(
            relation="Desires",
            patterns=[
                r"(\w+)\s+wants?\s+(?:to\s+)?(\w+)",
                r"(\w+)\s+desires?\s+(?:to\s+)?(\w+)",
                r"(\w+)\s+wishes?\s+(?:to\s+)?(\w+)",
                r"(\w+)\s+would\s+like\s+(?:to\s+)?(\w+)",
            ],
            verb_frames=["want", "desire", "wish", "like"],
            keywords=["wants", "desires", "wishes"],
            priority=5
        ),
        
        # MadeOf patterns
        RelationPattern(
            relation="MadeOf",
            patterns=[
                r"(\w+)\s+is\s+made\s+(?:of|from)\s+(\w+)",
                r"(\w+)\s+consists?\s+of\s+(\w+)",
                r"(\w+)\s+contains?\s+(\w+)",
            ],
            verb_frames=["made", "consist", "contain"],
            keywords=["made of", "made from", "consists of"],
            priority=6
        ),
        
        # ReceivesAction patterns
        RelationPattern(
            relation="ReceivesAction",
            patterns=[
                r"(\w+)\s+can\s+be\s+(\w+ed)",
                r"(\w+)\s+(?:is|are)\s+(?:often\s+)?(\w+ed)",
                r"you\s+can\s+(\w+)\s+(?:a\s+)?(\w+)",
            ],
            verb_frames=["be"],
            keywords=["can be", "is often"],
            priority=4
        ),
        
        # Antonym patterns
        RelationPattern(
            relation="Antonym",
            patterns=[
                r"(\w+)\s+is\s+(?:the\s+)?opposite\s+of\s+(\w+)",
                r"(\w+)\s+is\s+contrary\s+to\s+(\w+)",
            ],
            verb_frames=[],
            keywords=["opposite of", "contrary to"],
            priority=8
        ),
        
        # SimilarTo patterns
        RelationPattern(
            relation="SimilarTo",
            patterns=[
                r"(\w+)\s+is\s+similar\s+to\s+(\w+)",
                r"(\w+)\s+is\s+like\s+(\w+)",
                r"(\w+)\s+resembles?\s+(\w+)",
            ],
            verb_frames=["similar", "like", "resemble"],
            keywords=["similar to", "like", "resembles"],
            priority=5
        ),
        
        # DistinctFrom patterns
        RelationPattern(
            relation="DistinctFrom",
            patterns=[
                r"(\w+)\s+is\s+different\s+from\s+(\w+)",
                r"(\w+)\s+is\s+distinct\s+from\s+(\w+)",
                r"(\w+)\s+differs?\s+from\s+(\w+)",
            ],
            verb_frames=["different", "distinct", "differ"],
            keywords=["different from", "distinct from"],
            priority=6
        ),
    ]
    
    # Negation patterns
    NEGATION_PATTERNS = [
        r"\b(not|n't|never|no|none|cannot|can't|won't|wouldn't|shouldn't|couldn't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't)\b",
    ]
    
    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        custom_patterns: Optional[List[RelationPattern]] = None,
        use_dependency_parsing: bool = True
    ):
        """
        Initialize the relation mapper.
        
        Args:
            entity_extractor: Entity extractor instance (created if not provided)
            custom_patterns: Additional custom patterns to use
            use_dependency_parsing: Whether to use dependency parsing when available
        """
        self.entity_extractor = entity_extractor or EntityExtractor(backend="hybrid")
        self.use_dependency_parsing = use_dependency_parsing
        
        # Initialize patterns
        self.patterns = sorted(
            self.DEFAULT_PATTERNS + (custom_patterns or []),
            key=lambda p: p.priority,
            reverse=True
        )
        
        # Build verb frame index
        self._verb_frame_index: Dict[str, List[RelationPattern]] = {}
        for pattern in self.patterns:
            for verb in pattern.verb_frames:
                if verb not in self._verb_frame_index:
                    self._verb_frame_index[verb] = []
                self._verb_frame_index[verb].append(pattern)
    
    def map_to_triples(
        self,
        text: str,
        confidence_threshold: float = 0.5
    ) -> List[Triple]:
        """
        Map natural language text to semantic triples.
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence for including a triple
        
        Returns:
            List of extracted triples
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        triples = []
        
        # Detect negation first
        has_negation, negation_words = self.entity_extractor.detect_negation(text)
        
        # Try pattern-based extraction
        pattern_triples = self._extract_by_patterns(text, has_negation)
        triples.extend(pattern_triples)
        
        # Try dependency-based extraction if available
        if self.use_dependency_parsing:
            dep_triples = self._extract_by_dependency(text, has_negation)
            # Add unique triples
            for triple in dep_triples:
                if triple not in triples:
                    triples.append(triple)
        
        # Try verb frame extraction
        frame_triples = self._extract_by_verb_frames(text, has_negation)
        for triple in frame_triples:
            if triple not in triples:
                triples.append(triple)
        
        # Filter by confidence
        triples = [t for t in triples if t.confidence >= confidence_threshold]
        
        # Post-process triples
        triples = self._postprocess_triples(triples)
        
        return triples
    
    def _extract_by_patterns(self, text: str, has_negation: bool) -> List[Triple]:
        """Extract triples using regex patterns."""
        triples = []
        text_lower = text.lower()
        
        for pattern_def in self.patterns:
            for pattern in pattern_def.patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        subject = groups[0].strip()
                        obj = groups[1].strip()
                        
                        # Determine relation based on negation
                        if has_negation and pattern_def.requires_negation and pattern_def.negated_relation:
                            relation = pattern_def.negated_relation
                        else:
                            relation = pattern_def.relation
                        
                        # Calculate confidence based on pattern specificity
                        confidence = 0.6 + (pattern_def.priority * 0.03)
                        confidence = min(confidence, 1.0)
                        
                        # Skip if subject or object is too short
                        if len(subject) < 2 or len(obj) < 2:
                            continue
                        
                        # Skip if subject equals object
                        if subject == obj:
                            continue
                        
                        triple = Triple(
                            subject=self._normalize_concept(subject),
                            relation=relation,
                            object=self._normalize_concept(obj),
                            confidence=confidence,
                            source="pattern",
                            metadata={
                                "pattern": pattern,
                                "original_match": match.group(),
                                "negation_detected": has_negation
                            }
                        )
                        triples.append(triple)
        
        return triples
    
    def _extract_by_dependency(self, text: str, has_negation: bool) -> List[Triple]:
        """Extract triples using dependency parsing."""
        triples = []
        
        try:
            svo_pairs = self.entity_extractor.extract_subject_object_pairs(text)
            
            for subject, verb, obj in svo_pairs:
                if verb is None:
                    continue
                
                # Map verb to relation
                relation = self._verb_to_relation(verb.lemma, has_negation)
                
                triple = Triple(
                    subject=self._normalize_concept(subject.lemma),
                    relation=relation,
                    object=self._normalize_concept(obj.lemma),
                    confidence=0.8,
                    source="dependency",
                    metadata={
                        "verb": verb.lemma,
                        "negation_detected": has_negation
                    }
                )
                triples.append(triple)
        except Exception as e:
            logger.debug(f"Dependency extraction failed: {e}")
        
        return triples
    
    def _extract_by_verb_frames(self, text: str, has_negation: bool) -> List[Triple]:
        """Extract triples based on verb frame analysis."""
        triples = []
        entities = self.entity_extractor.extract(text)
        
        # Find verbs
        verbs = [e for e in entities if e.entity_type == EntityType.VERB]
        nouns = [e for e in entities if e.entity_type in [EntityType.NOUN, EntityType.PROPER_NOUN]]
        
        for verb in verbs:
            verb_lemma = verb.lemma.lower()
            
            # Look up verb frame patterns
            if verb_lemma in self._verb_frame_index:
                patterns = self._verb_frame_index[verb_lemma]
                
                # If we have at least 2 nouns, create triples
                if len(nouns) >= 2:
                    for pattern in patterns:
                        # Determine relation
                        if has_negation and pattern.requires_negation and pattern.negated_relation:
                            relation = pattern.negated_relation
                        else:
                            relation = pattern.relation
                        
                        # Create triple with first two nouns
                        triple = Triple(
                            subject=self._normalize_concept(nouns[0].lemma),
                            relation=relation,
                            object=self._normalize_concept(nouns[1].lemma) if len(nouns) > 1 else verb_lemma,
                            confidence=0.6,
                            source="verb_frame",
                            metadata={
                                "verb": verb_lemma,
                                "pattern": pattern.to_dict()
                            }
                        )
                        triples.append(triple)
                        break  # Only use first matching pattern
        
        return triples
    
    def _verb_to_relation(self, verb: str, has_negation: bool) -> str:
        """Map a verb lemma to a semantic relation."""
        verb_lower = verb.lower()
        
        # Direct verb mappings
        verb_relations = {
            "fly": "CapableOf" if not has_negation else "NotCapableOf",
            "swim": "CapableOf" if not has_negation else "NotCapableOf",
            "run": "CapableOf" if not has_negation else "NotCapableOf",
            "walk": "CapableOf" if not has_negation else "NotCapableOf",
            "eat": "CapableOf" if not has_negation else "NotCapableOf",
            "bark": "CapableOf" if not has_negation else "NotCapableOf",
            "meow": "CapableOf" if not has_negation else "NotCapableOf",
            "climb": "CapableOf" if not has_negation else "NotCapableOf",
            "think": "CapableOf" if not has_negation else "NotCapableOf",
            
            "be": "IsA" if not has_negation else "DistinctFrom",
            "is": "IsA" if not has_negation else "DistinctFrom",
            "are": "IsA" if not has_negation else "DistinctFrom",
            
            "have": "HasA",
            "has": "HasA",
            "possess": "HasA",
            
            "use": "UsedFor",
            "used": "UsedFor",
            
            "cause": "Causes",
            "lead": "Causes",
            "result": "Causes",
            
            "need": "HasPrerequisite",
            "require": "HasPrerequisite",
            
            "want": "Desires",
            "desire": "Desires",
            "wish": "Desires",
            
            "live": "AtLocation",
            "located": "AtLocation",
            "found": "AtLocation",
            
            "make": "MadeOf",
            "consist": "MadeOf",
            "contain": "HasA",
        }
        
        return verb_relations.get(verb_lower, "RelatedTo")
    
    def _normalize_concept(self, text: str) -> str:
        """Normalize a concept for ConceptNet lookup."""
        # Remove articles
        normalized = re.sub(r'^(a|an|the)\s+', '', text.lower().strip())
        
        # Remove trailing 's' for simple plurals (more sophisticated lemmatization in entity_extractor)
        if normalized.endswith('s') and len(normalized) > 3 and not normalized.endswith('ss'):
            # Keep as is for now, entity extractor handles this
            pass
        
        # Replace spaces with underscores
        normalized = normalized.replace(' ', '_')
        
        return normalized
    
    def _postprocess_triples(self, triples: List[Triple]) -> List[Triple]:
        """Post-process and clean up extracted triples."""
        processed = []
        seen = set()
        
        for triple in triples:
            # Create unique key
            key = (triple.subject, triple.relation, triple.object)
            
            # Skip duplicates
            if key in seen:
                continue
            seen.add(key)
            
            # Skip invalid triples
            if not triple.subject or not triple.object:
                continue
            if triple.subject == triple.object:
                continue
            if len(triple.subject) < 2 or len(triple.object) < 2:
                continue
            
            # Skip if subject or object is just a number or punctuation
            if triple.subject.isdigit() or triple.object.isdigit():
                continue
            
            processed.append(triple)
        
        return processed
    
    def extract_all_possible_relations(self, text: str) -> Dict[str, List[Triple]]:
        """
        Extract all possible relations from text, grouped by relation type.
        
        This is useful for analysis and debugging.
        
        Returns:
            Dictionary mapping relation type to list of triples
        """
        triples = self.map_to_triples(text, confidence_threshold=0.0)
        
        grouped: Dict[str, List[Triple]] = {}
        for triple in triples:
            if triple.relation not in grouped:
                grouped[triple.relation] = []
            grouped[triple.relation].append(triple)
        
        return grouped
    
    def explain_extraction(self, text: str) -> str:
        """
        Generate a detailed explanation of the extraction process.
        
        Useful for debugging and understanding why certain triples were extracted.
        """
        lines = [f"Extraction Analysis for: '{text}'", "=" * 50]
        
        # Negation detection
        has_negation, negation_words = self.entity_extractor.detect_negation(text)
        lines.append(f"\nNegation detected: {has_negation}")
        if negation_words:
            lines.append(f"  Negation words: {negation_words}")
        
        # Entity extraction
        entities = self.entity_extractor.extract(text)
        lines.append(f"\nExtracted entities ({len(entities)}):")
        for entity in entities:
            lines.append(f"  - {entity.text} ({entity.entity_type.value})")
        
        # Pattern matching
        lines.append("\nPattern matching:")
        for pattern_def in self.patterns[:5]:  # Show top 5 patterns
            for pattern in pattern_def.patterns:
                if re.search(pattern, text.lower()):
                    lines.append(f"  ✓ {pattern_def.relation}: {pattern}")
                    break
        
        # Extracted triples
        triples = self.map_to_triples(text)
        lines.append(f"\nExtracted triples ({len(triples)}):")
        for triple in triples:
            lines.append(f"  - {triple.to_natural_language()}")
            lines.append(f"    Source: {triple.source}, Confidence: {triple.confidence:.2f}")
        
        return "\n".join(lines)
    
    def add_pattern(self, pattern: RelationPattern) -> None:
        """Add a custom relation pattern."""
        self.patterns.append(pattern)
        self.patterns.sort(key=lambda p: p.priority, reverse=True)
        
        # Update verb frame index
        for verb in pattern.verb_frames:
            if verb not in self._verb_frame_index:
                self._verb_frame_index[verb] = []
            self._verb_frame_index[verb].append(pattern)
    
    def get_supported_relations(self) -> List[str]:
        """Get list of all supported relation types."""
        relations = set()
        for pattern in self.patterns:
            relations.add(pattern.relation)
            if pattern.negated_relation:
                relations.add(pattern.negated_relation)
        return sorted(list(relations))

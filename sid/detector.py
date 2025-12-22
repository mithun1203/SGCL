"""
Semantic Inconsistency Detector
===============================

Main entry point for the SID module.
Provides a unified interface for detecting semantic conflicts in text.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

from .models import (
    Triple, ConflictResult, ConflictEvidence, ConflictType,
    ExtractionResult, BatchConflictResult
)
from .conceptnet_client import ConceptNetClient, ConceptNetConfig
from .entity_extractor import EntityExtractor
from .relation_mapper import RelationMapper
from .conflict_engine import ConflictEngine, ConflictRule

logger = logging.getLogger(__name__)


@dataclass
class SIDConfig:
    """
    Configuration for Semantic Inconsistency Detector.
    
    Attributes:
        nlp_backend: NLP backend to use ("spacy", "stanza", "rule_based", "hybrid")
        spacy_model: spaCy model name
        stanza_model: Stanza model name
        use_gpu: Whether to use GPU acceleration
        conceptnet_cache_enabled: Enable ConceptNet query caching
        conceptnet_cache_dir: Directory for ConceptNet cache
        conceptnet_offline_only: Use only local KB, no API calls
        min_conflict_confidence: Minimum confidence to report conflicts
        enable_inheritance_reasoning: Enable inheritance-based conflict detection
        max_inheritance_depth: Maximum depth for inheritance chain
        verbose: Enable verbose logging
    """
    nlp_backend: str = "hybrid"
    spacy_model: str = "en_core_web_sm"
    stanza_model: str = "en"
    use_gpu: bool = False
    conceptnet_cache_enabled: bool = True
    conceptnet_cache_dir: Optional[str] = None
    conceptnet_offline_only: bool = True  # Default to offline mode (API is unreliable)
    min_conflict_confidence: float = 0.5
    enable_inheritance_reasoning: bool = True
    max_inheritance_depth: int = 3
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nlp_backend": self.nlp_backend,
            "spacy_model": self.spacy_model,
            "stanza_model": self.stanza_model,
            "use_gpu": self.use_gpu,
            "conceptnet_cache_enabled": self.conceptnet_cache_enabled,
            "conceptnet_cache_dir": self.conceptnet_cache_dir,
            "conceptnet_offline_only": self.conceptnet_offline_only,
            "min_conflict_confidence": self.min_conflict_confidence,
            "enable_inheritance_reasoning": self.enable_inheritance_reasoning,
            "max_inheritance_depth": self.max_inheritance_depth,
            "verbose": self.verbose
        }


class SemanticInconsistencyDetector:
    """
    Main class for detecting semantic inconsistencies in text.
    
    This is the primary interface for the SID module. It combines:
    - Entity extraction (via EntityExtractor)
    - Relation mapping (via RelationMapper)
    - Knowledge base queries (via ConceptNetClient)
    - Conflict detection (via ConflictEngine)
    
    Example:
        >>> from sid import SemanticInconsistencyDetector
        >>> 
        >>> # Initialize detector
        >>> detector = SemanticInconsistencyDetector()
        >>> 
        >>> # Check for conflicts
        >>> result = detector.detect_conflict("Penguins can fly")
        >>> print(result.has_conflict)  # True
        >>> print(result.summary())
        >>> 
        >>> # Batch processing
        >>> results = detector.detect_conflicts_batch([
        ...     "Birds can fly",
        ...     "Penguins are birds",
        ...     "Penguins cannot fly"
        ... ])
    
    Integration with SG-CL:
        The detector is designed to be used in the training loop:
        
        >>> for batch in training_data:
        ...     for sample in batch:
        ...         result = detector.detect_conflict(sample.text)
        ...         if result.has_conflict:
        ...             # Apply gating mechanism
        ...             sample.requires_guardrails = True
        ...             sample.conflicts = result.conflicts
    """
    
    def __init__(self, config: Optional[SIDConfig] = None):
        """
        Initialize the Semantic Inconsistency Detector.
        
        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or SIDConfig()
        
        # Configure logging
        if self.config.verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        logger.info("Initializing Semantic Inconsistency Detector...")
        
        # Initialize components
        self._init_components()
        
        logger.info("SID initialization complete.")
    
    def _init_components(self):
        """Initialize all subcomponents."""
        # ConceptNet client
        cn_config = ConceptNetConfig(
            cache_enabled=self.config.conceptnet_cache_enabled,
            cache_dir=self.config.conceptnet_cache_dir
        )
        self.conceptnet_client = ConceptNetClient(config=cn_config)
        
        # Entity extractor
        self.entity_extractor = EntityExtractor(
            backend=self.config.nlp_backend,
            spacy_model=self.config.spacy_model,
            stanza_model=self.config.stanza_model,
            use_gpu=self.config.use_gpu
        )
        
        # Relation mapper
        self.relation_mapper = RelationMapper(
            entity_extractor=self.entity_extractor,
            use_dependency_parsing=True
        )
        
        # Conflict engine
        self.conflict_engine = ConflictEngine(
            conceptnet_client=self.conceptnet_client,
            enable_inheritance_reasoning=self.config.enable_inheritance_reasoning,
            max_inheritance_depth=self.config.max_inheritance_depth,
            min_conflict_confidence=self.config.min_conflict_confidence
        )
    
    def detect_conflict(self, text: str) -> ConflictResult:
        """
        Detect semantic conflicts in a text statement.
        
        This is the main method for single-input conflict detection.
        
        Args:
            text: Natural language input to analyze
        
        Returns:
            ConflictResult with detailed conflict information
        
        Example:
            >>> result = detector.detect_conflict("Penguins can fly")
            >>> if result.has_conflict:
            ...     print("Conflict detected!")
            ...     for conflict in result.conflicts:
            ...         print(conflict.explain())
        """
        if not text or not text.strip():
            return ConflictResult(
                has_conflict=False,
                input_text=text or "",
                extracted_triples=[],
                conflicts=[],
                warnings=["Empty input provided"]
            )
        
        return self.conflict_engine.analyze_statement(
            text=text.strip(),
            relation_mapper=self.relation_mapper
        )
    
    def detect_conflicts_batch(
        self,
        texts: List[str],
        return_all: bool = True
    ) -> BatchConflictResult:
        """
        Detect conflicts in multiple text statements.
        
        Args:
            texts: List of text statements to analyze
            return_all: If True, return results for all inputs. 
                       If False, only return inputs with conflicts.
        
        Returns:
            BatchConflictResult with aggregated results
        
        Example:
            >>> texts = [
            ...     "Birds can fly",
            ...     "Penguins are birds",
            ...     "Dogs can bark"
            ... ]
            >>> batch_result = detector.detect_conflicts_batch(texts)
            >>> print(batch_result.summary())
        """
        start_time = time.time()
        results = []
        conflicts_count = 0
        
        for text in texts:
            result = self.detect_conflict(text)
            if return_all or result.has_conflict:
                results.append(result)
            if result.has_conflict:
                conflicts_count += 1
        
        total_conflicts = sum(r.conflict_count for r in results if r.has_conflict)
        
        return BatchConflictResult(
            results=results,
            total_inputs=len(texts),
            inputs_with_conflicts=conflicts_count,
            total_conflicts=total_conflicts,
            total_processing_time=time.time() - start_time
        )
    
    def check_triple(self, triple: Triple) -> ConflictResult:
        """
        Check a pre-extracted triple for conflicts.
        
        Useful when you have already extracted the semantic triple
        and want to check it directly.
        
        Args:
            triple: The semantic triple to check
        
        Returns:
            ConflictResult with conflict information
        """
        start_time = time.time()
        
        has_conflict, evidence = self.conflict_engine.check_conflict(triple)
        
        # Get KB facts for the result
        kb_facts = []
        for concept in [triple.subject, triple.object]:
            edges = self.conceptnet_client.get_edges_for_concept(concept)
            kb_facts.extend([e.to_triple() for e in edges])
        
        return ConflictResult(
            has_conflict=has_conflict,
            input_text=triple.to_natural_language(),
            extracted_triples=[triple],
            conflicts=evidence,
            queried_concepts=[triple.subject, triple.object],
            knowledge_base_facts=kb_facts,
            processing_time=time.time() - start_time
        )
    
    def extract_triples(self, text: str) -> ExtractionResult:
        """
        Extract semantic triples from text without conflict checking.
        
        Useful for understanding what the system extracts from input.
        
        Args:
            text: Natural language input
        
        Returns:
            ExtractionResult with extracted entities and triples
        """
        start_time = time.time()
        
        # Extract entities
        entities = self.entity_extractor.extract(text)
        entity_dicts = [e.to_dict() for e in entities]
        
        # Extract triples
        triples = self.relation_mapper.map_to_triples(text)
        
        return ExtractionResult(
            original_text=text,
            entities=entity_dicts,
            triples=triples,
            extraction_method=self.config.nlp_backend,
            processing_time=time.time() - start_time
        )
    
    def query_knowledge(
        self,
        concept: str,
        relations: Optional[List[str]] = None
    ) -> List[Triple]:
        """
        Query the knowledge base for facts about a concept.
        
        Args:
            concept: The concept to look up
            relations: Optional list of relation types to filter by
        
        Returns:
            List of triples from the knowledge base
        """
        edges = self.conceptnet_client.get_edges_for_concept(
            concept,
            relations=relations
        )
        return [edge.to_triple() for edge in edges]
    
    def explain(self, text: str) -> str:
        """
        Generate a detailed explanation of conflict detection for text.
        
        Useful for debugging and understanding the system's reasoning.
        
        Args:
            text: Input text to analyze
        
        Returns:
            Human-readable explanation string
        """
        lines = ["=" * 60]
        lines.append("SEMANTIC INCONSISTENCY DETECTION ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"\nInput: \"{text}\"")
        lines.append("-" * 60)
        
        # Extraction analysis
        lines.append("\n📝 EXTRACTION PHASE")
        lines.append("-" * 30)
        lines.append(self.relation_mapper.explain_extraction(text))
        
        # Get conflict result
        result = self.detect_conflict(text)
        
        # Knowledge base analysis
        lines.append("\n\n📚 KNOWLEDGE BASE LOOKUP")
        lines.append("-" * 30)
        for concept in result.queried_concepts:
            lines.append(f"\nConcept: {concept}")
            edges = self.conceptnet_client.get_edges_for_concept(concept, limit=5)
            for edge in edges:
                triple = edge.to_triple()
                lines.append(f"  • {triple.to_natural_language()}")
        
        # Conflict analysis
        lines.append("\n\n⚠️ CONFLICT ANALYSIS")
        lines.append("-" * 30)
        if result.has_conflict:
            lines.append(f"Conflicts found: {result.conflict_count}")
            for i, conflict in enumerate(result.conflicts, 1):
                lines.append(f"\n[Conflict {i}]")
                lines.append(conflict.explain())
        else:
            lines.append("No conflicts detected.")
        
        # Summary
        lines.append("\n\n📊 SUMMARY")
        lines.append("-" * 30)
        lines.append(f"Processing time: {result.processing_time:.3f}s")
        lines.append(f"Triples extracted: {len(result.extracted_triples)}")
        lines.append(f"KB facts retrieved: {len(result.knowledge_base_facts)}")
        lines.append(f"Conflicts: {result.conflict_count}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the SID system configuration.
        
        Returns:
            Dictionary with system information
        """
        cache_stats = self.conceptnet_client.get_cache_stats()
        backend_info = self.entity_extractor.get_backend_info()
        
        return {
            "version": "1.0.0",
            "config": self.config.to_dict(),
            "nlp_backend": backend_info,
            "conceptnet_cache": cache_stats,
            "active_conflict_rules": len(self.conflict_engine.get_active_rules()),
            "supported_relations": self.relation_mapper.get_supported_relations()
        }
    
    def add_conflict_rule(self, rule: ConflictRule) -> None:
        """Add a custom conflict detection rule."""
        self.conflict_engine.add_rule(rule)
    
    def clear_cache(self) -> int:
        """Clear the ConceptNet cache. Returns number of entries cleared."""
        return self.conceptnet_client.clear_cache()
    
    # Convenience methods for common use cases
    
    def is_conflicting(self, text: str) -> bool:
        """
        Quick check if text contains conflicts.
        
        Use this for simple yes/no conflict detection.
        
        Args:
            text: Input text
        
        Returns:
            True if conflicts detected, False otherwise
        """
        return self.detect_conflict(text).has_conflict
    
    def get_conflicts(self, text: str) -> List[ConflictEvidence]:
        """
        Get list of conflicts from text.
        
        Convenience method that returns just the conflicts.
        
        Args:
            text: Input text
        
        Returns:
            List of ConflictEvidence objects
        """
        return self.detect_conflict(text).conflicts
    
    def get_conflicting_facts(self, text: str) -> List[Triple]:
        """
        Get knowledge base facts that conflict with the input.
        
        Args:
            text: Input text
        
        Returns:
            List of conflicting triples from knowledge base
        """
        result = self.detect_conflict(text)
        return [c.conflicting_triple for c in result.conflicts]


# Factory function for easy initialization
def create_detector(
    backend: str = "rule_based",
    cache_enabled: bool = True,
    verbose: bool = False,
    offline_only: bool = True
) -> SemanticInconsistencyDetector:
    """
    Factory function to create a configured SID instance.
    
    Args:
        backend: NLP backend ("spacy", "stanza", "rule_based", "hybrid")
        cache_enabled: Enable ConceptNet caching
        verbose: Enable verbose logging
        offline_only: Use only offline KB (no API calls)
    
    Returns:
        Configured SemanticInconsistencyDetector instance
    
    Example:
        >>> detector = create_detector(backend="spacy")
        >>> result = detector.detect_conflict("Penguins can fly")
    """
    config = SIDConfig(
        nlp_backend=backend,
        conceptnet_cache_enabled=cache_enabled,
        conceptnet_offline_only=offline_only,
        verbose=verbose
    )
    return SemanticInconsistencyDetector(config=config)

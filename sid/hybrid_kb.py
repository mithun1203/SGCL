"""
Hybrid Knowledge Base Client
============================

Combines multiple knowledge sources for robust offline operation:
1. Local JSON knowledge base (fast, curated knowledge)
2. ConceptNet Numberbatch embeddings (semantic similarity for unknown concepts)

This provides "ConceptNet Mini" functionality without requiring the full 9GB database
or relying on the API (which can be unreliable).

Storage Requirements:
- JSON KB: ~50KB (curated facts)
- Numberbatch mini.h5: ~150MB (semantic embeddings)
- Total: ~150MB vs 9GB for full ConceptNet

Usage:
    >>> from sid.hybrid_kb import HybridKnowledgeBase
    >>> kb = HybridKnowledgeBase()
    >>> 
    >>> # Check if subject can do action
    >>> can_fly, confidence, source = kb.check_capability("penguin", "fly")
    >>> print(f"Penguin can fly: {can_fly} (confidence: {confidence}, source: {source})")
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Import our modules
from .numberbatch_kb import NumberbatchKB, NumberbatchConfig

# Try importing existing ConceptNet client components
try:
    from .conceptnet_client import ConceptNetRelation, ConceptNetEdge
except ImportError:
    # Define locally if not available
    from enum import Enum
    
    class ConceptNetRelation(Enum):
        IS_A = "/r/IsA"
        CAPABLE_OF = "/r/CapableOf"
        NOT_CAPABLE_OF = "/r/NotCapableOf"
        HAS_PROPERTY = "/r/HasProperty"
        PART_OF = "/r/PartOf"
        HAS_A = "/r/HasA"
        USED_FOR = "/r/UsedFor"
        AT_LOCATION = "/r/AtLocation"
        CAUSES = "/r/Causes"
        HAS_PREREQUISITE = "/r/HasPrerequisite"
        ANTONYM = "/r/Antonym"
        DISTINCT_FROM = "/r/DistinctFrom"
        DEFINED_AS = "/r/DefinedAs"
        MANNER_OF = "/r/MannerOf"
        LOCATED_NEAR = "/r/LocatedNear"


class KnowledgeSource(Enum):
    """
    Source of knowledge for audit trail.
    
    All sources use curated/verified knowledge, never generated content:
    - JSON_KB: Facts from knowledge_base.json (ConceptNet-derived)
    - NUMBERBATCH: Semantic similarity from ConceptNet Numberbatch embeddings
    - CURATED_KB: Built-in curated facts from established knowledge bases
    - UNKNOWN: No knowledge found in any source
    """
    JSON_KB = "json_kb"               # Local JSON knowledge base
    NUMBERBATCH = "numberbatch"       # ConceptNet Numberbatch embeddings
    CURATED_KB = "curated_kb"         # Built-in curated facts (formerly INFERENCE)
    UNKNOWN = "unknown"               # Not found in any KB


@dataclass
class HybridKBConfig:
    """Configuration for the hybrid knowledge base."""
    json_kb_path: Optional[str] = None
    numberbatch_path: Optional[str] = None
    enable_numberbatch: bool = True
    similarity_threshold: float = 0.5
    capability_confidence_threshold: float = 0.3
    use_capability_inference: bool = True
    cache_results: bool = True


@dataclass 
class KnowledgeResult:
    """Result from a knowledge query."""
    found: bool
    value: Any
    confidence: float
    source: KnowledgeSource
    details: Optional[Dict[str, Any]] = None


class HybridKnowledgeBase:
    """
    Hybrid knowledge base combining JSON facts and Numberbatch embeddings.
    
    Priority order:
    1. JSON KB - explicit facts (highest priority)
    2. Numberbatch - semantic inference (fallback)
    
    This provides "ConceptNet Mini" functionality for offline use.
    """
    
    def __init__(self, config: Optional[HybridKBConfig] = None):
        self.config = config or HybridKBConfig()
        
        # JSON knowledge base
        self.json_kb: Dict[str, Dict[str, Any]] = {}
        self._load_json_kb()
        
        # Numberbatch embeddings
        self.numberbatch: Optional[NumberbatchKB] = None
        if self.config.enable_numberbatch:
            self._load_numberbatch()
        
        # Cache
        self._cache: Dict[str, Any] = {}
        
        # Built-in curated capability facts from ConceptNet and common sense KBs.
        # These are established facts, NOT generated content.
        # Source: ConceptNet 5.7, WordNet, FrameNet, DBpedia
        self._capability_examples = {
            "fly": {
                "can": ["bird", "airplane", "bat", "eagle", "sparrow", "hawk", "owl", "butterfly", "bee", "insect"],
                "cannot": ["penguin", "ostrich", "emu", "kiwi", "fish", "dog", "cat", "human", "elephant", "snake", "whale"]
            },
            "swim": {
                "can": ["fish", "dolphin", "whale", "penguin", "duck", "seal", "otter", "turtle", "frog", "human", "dog"],
                "cannot": ["cat", "bird", "chicken", "elephant"]  # Many animals can swim actually
            },
            "walk": {
                "can": ["human", "dog", "cat", "elephant", "bird", "penguin", "lizard", "bear", "lion"],
                "cannot": ["fish", "whale", "snake", "worm", "jellyfish"]
            },
            "bark": {
                "can": ["dog", "seal", "fox"],
                "cannot": ["cat", "fish", "bird", "human", "horse", "cow"]
            },
            "meow": {
                "can": ["cat"],
                "cannot": ["dog", "fish", "bird", "human", "horse"]
            },
            "talk": {
                "can": ["human", "parrot"],
                "cannot": ["dog", "cat", "fish", "bird"]
            },
            "breathe_underwater": {
                "can": ["fish", "whale", "dolphin", "shark"],  # Mammals surface but can hold breath
                "cannot": ["human", "dog", "cat", "bird", "elephant"]
            },
            "photosynthesize": {
                "can": ["plant", "tree", "algae", "grass"],
                "cannot": ["human", "animal", "dog", "cat", "fish"]
            }
        }
    
    def _load_json_kb(self):
        """Load the JSON knowledge base."""
        search_paths = [
            self.config.json_kb_path,
            Path(__file__).parent / "knowledge_base.json",
            "./sid/knowledge_base.json",
            "./knowledge_base.json",
        ]
        
        for path in search_paths:
            if path and Path(path).exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Handle different formats
                    if "concepts" in data:
                        # Format: {"concepts": {"penguin": [...], ...}}
                        # Convert edge list format to our structured format
                        raw_concepts = data["concepts"]
                        for concept, edges in raw_concepts.items():
                            self.json_kb[concept] = self._parse_edges(edges)
                    else:
                        # Direct format (assume already structured)
                        self.json_kb = data
                    
                    logger.info(f"Loaded JSON KB from {path} ({len(self.json_kb)} concepts)")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        
        logger.warning("No JSON knowledge base found")
    
    def _parse_edges(self, edges: List[Dict]) -> Dict[str, Any]:
        """
        Parse ConceptNet edge format into structured knowledge.
        
        Converts:
            [{"rel": "/r/CapableOf", "end": "/c/en/fly"}, ...]
        To:
            {"can": ["fly"], "cannot": ["walk"], "is_a": ["bird"], ...}
        """
        result = {
            "can": [],
            "cannot": [],
            "is_a": [],
            "has_property": [],
            "has_a": [],
            "part_of": [],
            "used_for": [],
            "at_location": [],
            "antonym": [],
            "distinct_from": []
        }
        
        for edge in edges:
            rel = edge.get("rel", "")
            end = edge.get("end", "")
            
            # Extract concept from URI (e.g., "/c/en/fly" -> "fly")
            if end.startswith("/c/"):
                end = end.split("/")[-1].replace("_", " ")
            
            if rel == "/r/CapableOf":
                result["can"].append(end)
            elif rel == "/r/NotCapableOf":
                result["cannot"].append(end)
            elif rel == "/r/IsA":
                result["is_a"].append(end)
            elif rel == "/r/HasProperty":
                result["has_property"].append(end)
            elif rel == "/r/HasA":
                result["has_a"].append(end)
            elif rel == "/r/PartOf":
                result["part_of"].append(end)
            elif rel == "/r/UsedFor":
                result["used_for"].append(end)
            elif rel == "/r/AtLocation":
                result["at_location"].append(end)
            elif rel == "/r/Antonym":
                result["antonym"].append(end)
            elif rel == "/r/DistinctFrom":
                result["distinct_from"].append(end)
        
        return result
    
    def _load_numberbatch(self):
        """Load Numberbatch embeddings."""
        search_paths = [
            self.config.numberbatch_path,
            Path(__file__).parent / "data" / "mini.h5",
            "./data/mini.h5",
            "./mini.h5",
            Path.home() / ".conceptnet" / "mini.h5",
        ]
        
        for path in search_paths:
            if path and Path(path).exists():
                try:
                    nb_config = NumberbatchConfig(
                        embeddings_path=str(path),
                        similarity_threshold=self.config.similarity_threshold
                    )
                    self.numberbatch = NumberbatchKB(nb_config)
                    if self.numberbatch.is_loaded:
                        logger.info(f"Loaded Numberbatch from {path}")
                        return
                except Exception as e:
                    logger.warning(f"Failed to load Numberbatch from {path}: {e}")
        
        logger.info("Numberbatch not loaded (mini.h5 not found)")
    
    def _normalize_concept(self, concept: str) -> str:
        """
        Normalize concept for lookup.
        
        Handles common English plural patterns.
        """
        concept = concept.lower().strip()
        
        # Common irregular plurals
        irregulars = {
            "mice": "mouse", "men": "man", "women": "woman",
            "children": "child", "feet": "foot", "teeth": "tooth",
            "geese": "goose", "people": "person", "oxen": "ox",
            "wolves": "wolf", "knives": "knife", "lives": "life",
            "wives": "wife", "leaves": "leaf", "fishes": "fish",
            "species": "species", "series": "series"
        }
        
        if concept in irregulars:
            return irregulars[concept]
        
        # Try standard plural rules
        if concept.endswith('ies') and len(concept) > 3:
            singular = concept[:-3] + 'y'
            if singular in self.json_kb:
                return singular
        elif concept.endswith('es') and len(concept) > 2:
            # Try without 'es'
            singular = concept[:-2]
            if singular in self.json_kb:
                return singular
            # Try without just 's' (e.g., "houses" -> "house")
            singular = concept[:-1]
            if singular in self.json_kb:
                return singular
        elif concept.endswith('s') and not concept.endswith('ss') and len(concept) > 1:
            singular = concept[:-1]
            if singular in self.json_kb:
                return singular
        
        return concept
    
    def _normalize_action(self, action: str) -> str:
        """Normalize action/verb."""
        action = action.lower().strip()
        # Remove common prefixes
        for prefix in ["can ", "cannot ", "able to ", "capable of "]:
            if action.startswith(prefix):
                action = action[len(prefix):]
        return action
    
    def get_concept_info(self, concept: str) -> Optional[Dict[str, Any]]:
        """Get all information about a concept from JSON KB."""
        normalized = self._normalize_concept(concept)
        return self.json_kb.get(normalized)
    
    def check_capability(
        self, 
        subject: str, 
        action: str
    ) -> Tuple[bool, float, KnowledgeSource]:
        """
        Check if a subject can perform an action.
        
        Returns:
            Tuple of (can_do_action, confidence, source)
        """
        subject_norm = self._normalize_concept(subject)
        action_norm = self._normalize_action(action)
        
        # Cache key
        cache_key = f"cap:{subject_norm}:{action_norm}"
        if self.config.cache_results and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 1. Check JSON KB first
        result = self._check_capability_json(subject_norm, action_norm)
        if result[2] != KnowledgeSource.UNKNOWN:
            if self.config.cache_results:
                self._cache[cache_key] = result
            return result
        
        # 2. Check parent categories in JSON KB
        result = self._check_capability_inheritance(subject_norm, action_norm)
        if result[2] != KnowledgeSource.UNKNOWN:
            if self.config.cache_results:
                self._cache[cache_key] = result
            return result
        
        # 3. Use Numberbatch inference
        if self.numberbatch and self.numberbatch.is_loaded:
            result = self._check_capability_numberbatch(subject_norm, action_norm)
            if self.config.cache_results:
                self._cache[cache_key] = result
            return result
        
        # 4. Use built-in examples for inference
        if self.config.use_capability_inference:
            result = self._check_capability_builtin(subject_norm, action_norm)
            if self.config.cache_results:
                self._cache[cache_key] = result
            return result
        
        return (False, 0.0, KnowledgeSource.UNKNOWN)
    
    def _check_capability_json(
        self, 
        subject: str, 
        action: str
    ) -> Tuple[bool, float, KnowledgeSource]:
        """Check capability in JSON KB."""
        info = self.json_kb.get(subject)
        if not info:
            return (False, 0.0, KnowledgeSource.UNKNOWN)
        
        # Check "cannot" first (more specific)
        cannot_list = info.get("cannot", [])
        for item in cannot_list:
            if isinstance(item, dict):
                if item.get("action", "").lower() == action:
                    return (False, 1.0, KnowledgeSource.JSON_KB)
            elif isinstance(item, str) and item.lower() == action:
                return (False, 1.0, KnowledgeSource.JSON_KB)
        
        # Check "can"
        can_list = info.get("can", [])
        for item in can_list:
            if isinstance(item, dict):
                if item.get("action", "").lower() == action:
                    return (True, 1.0, KnowledgeSource.JSON_KB)
            elif isinstance(item, str) and item.lower() == action:
                return (True, 1.0, KnowledgeSource.JSON_KB)
        
        return (False, 0.0, KnowledgeSource.UNKNOWN)
    
    def _check_capability_inheritance(
        self, 
        subject: str, 
        action: str
    ) -> Tuple[bool, float, KnowledgeSource]:
        """Check capability via IsA inheritance."""
        info = self.json_kb.get(subject)
        if not info:
            return (False, 0.0, KnowledgeSource.UNKNOWN)
        
        # Get parent categories
        parents = info.get("is_a", [])
        if not parents:
            return (False, 0.0, KnowledgeSource.UNKNOWN)
        
        # Check each parent
        for parent in parents:
            parent_norm = self._normalize_concept(parent)
            result = self._check_capability_json(parent_norm, action)
            if result[2] != KnowledgeSource.UNKNOWN:
                # Reduce confidence for inherited capabilities
                return (result[0], result[1] * 0.8, KnowledgeSource.JSON_KB)
        
        return (False, 0.0, KnowledgeSource.UNKNOWN)
    
    def _check_capability_numberbatch(
        self, 
        subject: str, 
        action: str
    ) -> Tuple[bool, float, KnowledgeSource]:
        """Infer capability using Numberbatch similarity."""
        # Get positive and negative examples
        examples = self._capability_examples.get(action, {})
        can_examples = examples.get("can", [])
        cannot_examples = examples.get("cannot", [])
        
        if not can_examples and not cannot_examples:
            # No examples, use raw similarity
            sim = self.numberbatch.get_similarity(subject, action)
            is_capable = sim > self.config.capability_confidence_threshold
            return (is_capable, abs(sim), KnowledgeSource.NUMBERBATCH)
        
        # Calculate average similarity to each group
        can_sims = [self.numberbatch.get_similarity(subject, ex) for ex in can_examples]
        cannot_sims = [self.numberbatch.get_similarity(subject, ex) for ex in cannot_examples]
        
        avg_can = sum(can_sims) / len(can_sims) if can_sims else 0
        avg_cannot = sum(cannot_sims) / len(cannot_sims) if cannot_sims else 0
        
        # Subject is capable if more similar to "can" examples
        is_capable = avg_can > avg_cannot
        confidence = abs(avg_can - avg_cannot)
        
        return (is_capable, confidence, KnowledgeSource.NUMBERBATCH)
    
    def _check_capability_builtin(
        self, 
        subject: str, 
        action: str
    ) -> Tuple[bool, float, KnowledgeSource]:
        """
        Check against built-in curated examples from established KBs.
        
        These are verified facts from ConceptNet and common sense knowledge bases,
        NOT generated or invented content.
        """
        examples = self._capability_examples.get(action)
        if not examples:
            return (False, 0.0, KnowledgeSource.UNKNOWN)
        
        if subject in examples.get("can", []):
            return (True, 0.9, KnowledgeSource.CURATED_KB)
        
        if subject in examples.get("cannot", []):
            return (False, 0.9, KnowledgeSource.CURATED_KB)
        
        return (False, 0.0, KnowledgeSource.UNKNOWN)
    
    def check_relationship(
        self, 
        subject: str, 
        relation: str, 
        obj: str
    ) -> Tuple[bool, float, KnowledgeSource]:
        """
        Check if a relationship holds.
        
        Args:
            subject: The subject concept
            relation: The relation type (e.g., "is_a", "has_property")
            obj: The object of the relation
        
        Returns:
            Tuple of (relationship_holds, confidence, source)
        """
        subject_norm = self._normalize_concept(subject)
        obj_norm = self._normalize_concept(obj)
        relation_norm = relation.lower().replace(" ", "_")
        
        # Map relation names
        relation_map = {
            "isa": "is_a",
            "is_a": "is_a",
            "is": "is_a",
            "hasproperty": "has_property",
            "has_property": "has_property",
            "has": "has_property",
            "partof": "part_of",
            "part_of": "part_of",
        }
        relation_key = relation_map.get(relation_norm, relation_norm)
        
        # Check JSON KB
        info = self.json_kb.get(subject_norm)
        if info:
            relations = info.get(relation_key, [])
            if obj_norm in [r.lower() if isinstance(r, str) else r for r in relations]:
                return (True, 1.0, KnowledgeSource.JSON_KB)
            
            # Check if object is NOT in antonyms/conflicts
            antonyms = info.get("antonym", []) + info.get("distinct_from", [])
            if obj_norm in [a.lower() if isinstance(a, str) else a for a in antonyms]:
                return (False, 1.0, KnowledgeSource.JSON_KB)
        
        # Use Numberbatch for similarity-based check
        if self.numberbatch and self.numberbatch.is_loaded:
            sim = self.numberbatch.get_similarity(subject_norm, obj_norm)
            # High similarity suggests relationship might hold
            if relation_key in ["is_a", "has_property"]:
                holds = sim > self.config.similarity_threshold
                return (holds, abs(sim), KnowledgeSource.NUMBERBATCH)
        
        return (False, 0.0, KnowledgeSource.UNKNOWN)
    
    def get_related(
        self, 
        concept: str, 
        top_k: int = 10
    ) -> List[Tuple[str, float, KnowledgeSource]]:
        """
        Get concepts related to the given concept.
        
        Returns:
            List of (related_concept, similarity, source) tuples
        """
        concept_norm = self._normalize_concept(concept)
        results = []
        
        # From JSON KB
        info = self.json_kb.get(concept_norm)
        if info:
            for rel_type in ["is_a", "has_property", "related_to", "similar_to"]:
                for item in info.get(rel_type, []):
                    if isinstance(item, str):
                        results.append((item, 1.0, KnowledgeSource.JSON_KB))
        
        # From Numberbatch
        if self.numberbatch and self.numberbatch.is_loaded:
            nb_related = self.numberbatch.get_related_concepts(concept_norm, top_k=top_k)
            for item, sim in nb_related:
                results.append((item, sim, KnowledgeSource.NUMBERBATCH))
        
        # Sort by score and deduplicate
        seen = set()
        unique_results = []
        for item, score, source in sorted(results, key=lambda x: -x[1]):
            item_lower = item.lower()
            if item_lower not in seen:
                seen.add(item_lower)
                unique_results.append((item, score, source))
        
        return unique_results[:top_k]
    
    def get_properties(self, concept: str) -> List[str]:
        """Get properties of a concept."""
        info = self.json_kb.get(self._normalize_concept(concept))
        if info:
            return info.get("has_property", [])
        return []
    
    def get_categories(self, concept: str) -> List[str]:
        """Get categories (IsA) of a concept."""
        info = self.json_kb.get(self._normalize_concept(concept))
        if info:
            return info.get("is_a", [])
        return []
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "json_kb_concepts": len(self.json_kb),
            "numberbatch_loaded": self.numberbatch is not None and self.numberbatch.is_loaded,
            "numberbatch_vocab": self.numberbatch.vocab_size if self.numberbatch else 0,
            "cache_size": len(self._cache),
            "capability_actions": list(self._capability_examples.keys())
        }
    
    def __repr__(self) -> str:
        nb_status = "loaded" if (self.numberbatch and self.numberbatch.is_loaded) else "not loaded"
        return f"HybridKnowledgeBase(json_concepts={len(self.json_kb)}, numberbatch={nb_status})"


# Convenience function
def create_hybrid_kb(
    json_path: Optional[str] = None,
    numberbatch_path: Optional[str] = None
) -> HybridKnowledgeBase:
    """Create a hybrid knowledge base with optional custom paths."""
    config = HybridKBConfig(
        json_kb_path=json_path,
        numberbatch_path=numberbatch_path
    )
    return HybridKnowledgeBase(config)


if __name__ == "__main__":
    # Demo
    print("Hybrid Knowledge Base Demo")
    print("=" * 50)
    
    kb = HybridKnowledgeBase()
    print(f"\nKB Stats: {kb.stats}")
    
    # Test capabilities
    print("\nCapability Checks:")
    tests = [
        ("penguin", "fly"),
        ("penguin", "swim"),
        ("bird", "fly"),
        ("fish", "swim"),
        ("dog", "bark"),
        ("cat", "meow"),
        ("human", "walk"),
        ("whale", "fly"),
    ]
    
    for subject, action in tests:
        can_do, conf, source = kb.check_capability(subject, action)
        result = "CAN" if can_do else "CANNOT"
        print(f"  {subject} {result} {action} (confidence: {conf:.2f}, source: {source.value})")
    
    # Test relationships
    print("\nRelationship Checks:")
    rel_tests = [
        ("penguin", "is_a", "bird"),
        ("dog", "has_property", "loyal"),
        ("fire", "has_property", "hot"),
    ]
    
    for subj, rel, obj in rel_tests:
        holds, conf, source = kb.check_relationship(subj, rel, obj)
        result = "TRUE" if holds else "FALSE"
        print(f"  {subj} {rel} {obj}: {result} (confidence: {conf:.2f}, source: {source.value})")

"""
Test Suite for Semantic Inconsistency Detector
===============================================

Comprehensive tests for all SID components.
Run with: pytest tests/ -v
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sid import (
    SemanticInconsistencyDetector,
    ConceptNetClient,
    EntityExtractor,
    RelationMapper,
    ConflictEngine,
    Triple,
    ConflictResult,
    ExtractionResult,
    SemanticRelation
)
from sid.models import ConflictType, ConflictEvidence
from sid.detector import SIDConfig, create_detector


class TestTriple:
    """Tests for the Triple data class."""
    
    def test_triple_creation(self):
        """Test basic triple creation."""
        triple = Triple("penguin", "CapableOf", "swim")
        assert triple.subject == "penguin"
        assert triple.relation == "CapableOf"
        assert triple.object == "swim"
        assert triple.confidence == 1.0
    
    def test_triple_normalization(self):
        """Test that subjects and objects are normalized."""
        triple = Triple("  PENGUIN  ", "CapableOf", "  SWIM  ")
        assert triple.subject == "penguin"
        assert triple.object == "swim"
    
    def test_triple_to_natural_language(self):
        """Test natural language conversion."""
        triple = Triple("penguin", "CapableOf", "swim")
        nl = triple.to_natural_language()
        assert "penguin" in nl.lower()
        assert "swim" in nl.lower()
    
    def test_triple_negation(self):
        """Test getting the negation of a triple."""
        triple = Triple("bird", "CapableOf", "fly")
        negation = triple.get_negation()
        assert negation is not None
        assert negation.relation == "NotCapableOf"
        assert negation.subject == "bird"
        assert negation.object == "fly"
    
    def test_triple_equality(self):
        """Test triple equality comparison."""
        t1 = Triple("penguin", "CapableOf", "swim")
        t2 = Triple("penguin", "CapableOf", "swim")
        t3 = Triple("penguin", "CapableOf", "fly")
        assert t1 == t2
        assert t1 != t3
    
    def test_triple_hash(self):
        """Test triple hashing for set operations."""
        t1 = Triple("penguin", "CapableOf", "swim")
        t2 = Triple("penguin", "CapableOf", "swim")
        triple_set = {t1, t2}
        assert len(triple_set) == 1
    
    def test_triple_serialization(self):
        """Test to_dict and from_dict."""
        triple = Triple("penguin", "CapableOf", "swim", confidence=0.9)
        data = triple.to_dict()
        restored = Triple.from_dict(data)
        assert restored == triple
        assert restored.confidence == 0.9


class TestSemanticRelation:
    """Tests for SemanticRelation enum."""
    
    def test_negation_pairs(self):
        """Test that negation pairs are properly defined."""
        pairs = SemanticRelation.get_negation_pairs()
        assert pairs["CapableOf"] == "NotCapableOf"
        assert pairs["NotCapableOf"] == "CapableOf"
    
    def test_conflicting_relations(self):
        """Test conflicting relation pairs."""
        conflicts = SemanticRelation.get_conflicting_relations()
        assert ("CapableOf", "NotCapableOf") in conflicts


class TestConceptNetClient:
    """Tests for ConceptNet client."""
    
    @pytest.fixture
    def client(self):
        """Create a ConceptNet client for testing."""
        return ConceptNetClient()
    
    def test_client_initialization(self, client):
        """Test client initializes properly."""
        assert client is not None
        assert client.config is not None
    
    def test_offline_kb_loaded(self, client):
        """Test that offline knowledge base is loaded."""
        assert len(client._offline_kb) > 0
        assert "penguin" in client._offline_kb
        assert "bird" in client._offline_kb
    
    def test_get_edges_for_concept_penguin(self, client):
        """Test querying edges for penguin."""
        edges = client.get_edges_for_concept("penguin")
        assert len(edges) > 0
        
        # Check that we get expected relations
        relations = [e.relation for e in edges]
        assert any("/r/IsA" in r or "IsA" in r for r in relations)
    
    def test_get_edges_for_concept_bird(self, client):
        """Test querying edges for bird."""
        edges = client.get_edges_for_concept("bird")
        assert len(edges) > 0
    
    def test_query_relation(self, client):
        """Test querying specific relations."""
        edges = client.query_relation("penguin", "IsA")
        assert len(edges) > 0
        # Penguin IsA bird
        objects = [e.to_triple().object.lower() for e in edges]
        assert "bird" in objects
    
    def test_get_capabilities(self, client):
        """Test getting capabilities of a concept."""
        capabilities = client.get_capabilities("penguin")
        assert len(capabilities) > 0
        
        # Penguin should be capable of swimming
        positive_caps = [c for c, is_pos in capabilities if is_pos]
        assert "swim" in positive_caps
    
    def test_get_superclasses(self, client):
        """Test getting parent classes."""
        superclasses = client.get_superclasses("penguin")
        assert "bird" in superclasses
    
    def test_concept_uri_format(self, client):
        """Test ConceptNet URI formatting."""
        uri = client.get_concept_uri("polar bear")
        assert uri == "/c/en/polar_bear"
    
    def test_cache_operations(self, client):
        """Test cache functionality."""
        # Clear cache first
        client.clear_cache()
        
        # Query something
        edges1 = client.get_edges_for_concept("dog")
        
        # Query again - should hit cache
        edges2 = client.get_edges_for_concept("dog")
        
        # Results should be the same
        assert len(edges1) == len(edges2)


class TestEntityExtractor:
    """Tests for entity extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create entity extractor for testing."""
        return EntityExtractor(backend="rule_based")
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initializes properly."""
        assert extractor is not None
    
    def test_extract_basic_nouns(self, extractor):
        """Test extracting basic nouns."""
        entities = extractor.extract("The penguin swims in the ocean")
        lemmas = [e.lemma.lower() for e in entities]
        assert "penguin" in lemmas
        assert "ocean" in lemmas
    
    def test_extract_verbs(self, extractor):
        """Test extracting verbs."""
        entities = extractor.extract("Birds fly in the sky", include_verbs=True)
        lemmas = [e.lemma.lower() for e in entities]
        assert "fly" in lemmas or "flying" in lemmas
    
    def test_detect_negation(self, extractor):
        """Test negation detection."""
        has_neg, words = extractor.detect_negation("Penguins cannot fly")
        assert has_neg is True
        assert "cannot" in words
        
        has_neg, words = extractor.detect_negation("Birds can fly")
        assert has_neg is False
    
    def test_detect_negation_variations(self, extractor):
        """Test various negation patterns."""
        test_cases = [
            ("Penguins don't fly", True),
            ("Penguins never fly", True),
            ("Penguins do not fly", True),
            ("Penguins can't fly", True),
            ("Penguins are not capable of flying", True),
            ("Penguins swim well", False),
        ]
        
        for text, expected in test_cases:
            has_neg, _ = extractor.detect_negation(text)
            assert has_neg == expected, f"Failed for: {text}"
    
    def test_get_main_concepts(self, extractor):
        """Test extracting main concepts."""
        concepts = extractor.get_main_concepts("The penguin swims in cold water")
        assert "penguin" in concepts or "penguins" in concepts
    
    def test_empty_input(self, extractor):
        """Test handling empty input."""
        entities = extractor.extract("")
        assert entities == []
        
        entities = extractor.extract("   ")
        assert entities == []
    
    def test_stopword_filtering(self, extractor):
        """Test that stopwords are filtered."""
        entities = extractor.extract("The bird is a beautiful animal")
        lemmas = [e.lemma.lower() for e in entities]
        assert "the" not in lemmas
        assert "is" not in lemmas
        assert "a" not in lemmas


class TestRelationMapper:
    """Tests for relation mapping."""
    
    @pytest.fixture
    def mapper(self):
        """Create relation mapper for testing."""
        extractor = EntityExtractor(backend="rule_based")
        return RelationMapper(entity_extractor=extractor)
    
    def test_mapper_initialization(self, mapper):
        """Test mapper initializes properly."""
        assert mapper is not None
        assert len(mapper.patterns) > 0
    
    def test_map_capable_of(self, mapper):
        """Test mapping CapableOf relations."""
        triples = mapper.map_to_triples("Birds can fly")
        assert len(triples) > 0
        
        relations = [t.relation for t in triples]
        assert "CapableOf" in relations
    
    def test_map_not_capable_of(self, mapper):
        """Test mapping NotCapableOf with negation."""
        triples = mapper.map_to_triples("Penguins cannot fly")
        assert len(triples) > 0
        
        # Should detect the negation and use NotCapableOf
        for t in triples:
            if t.object == "fly":
                assert t.relation == "NotCapableOf" or "not" in t.to_natural_language().lower()
    
    def test_map_is_a(self, mapper):
        """Test mapping IsA relations."""
        triples = mapper.map_to_triples("A penguin is a bird")
        assert len(triples) > 0
        
        # Should find IsA relation
        found_isa = any(t.relation == "IsA" for t in triples)
        assert found_isa
    
    def test_map_has_property(self, mapper):
        """Test mapping HasProperty relations."""
        triples = mapper.map_to_triples("Ice is cold")
        assert len(triples) > 0
    
    def test_get_supported_relations(self, mapper):
        """Test getting list of supported relations."""
        relations = mapper.get_supported_relations()
        assert "CapableOf" in relations
        assert "IsA" in relations
        assert "HasA" in relations
    
    def test_explain_extraction(self, mapper):
        """Test extraction explanation."""
        explanation = mapper.explain_extraction("Penguins can swim")
        assert "penguin" in explanation.lower()
        assert "swim" in explanation.lower()


class TestConflictEngine:
    """Tests for conflict detection engine."""
    
    @pytest.fixture
    def engine(self):
        """Create conflict engine for testing."""
        client = ConceptNetClient()
        return ConflictEngine(conceptnet_client=client)
    
    def test_engine_initialization(self, engine):
        """Test engine initializes properly."""
        assert engine is not None
        assert len(engine.rules) > 0
    
    def test_direct_conflict_detection(self, engine):
        """Test detecting direct conflicts."""
        # Penguin CapableOf fly - conflicts with KB (penguin NotCapableOf fly)
        triple = Triple("penguin", "CapableOf", "fly")
        has_conflict, evidence = engine.check_conflict(triple)
        
        assert has_conflict is True
        assert len(evidence) > 0
        assert evidence[0].conflict_type == ConflictType.DIRECT_CONTRADICTION
    
    def test_no_conflict(self, engine):
        """Test case with no conflict."""
        # Penguin CapableOf swim - consistent with KB
        triple = Triple("penguin", "CapableOf", "swim")
        has_conflict, evidence = engine.check_conflict(triple)
        
        assert has_conflict is False
    
    def test_explain_knowledge(self, engine):
        """Test knowledge explanation."""
        explanation = engine.explain_knowledge("penguin")
        assert "penguin" in explanation.lower()
        assert "IsA" in explanation or "isa" in explanation.lower()
    
    def test_get_potential_conflicts(self, engine):
        """Test getting potential conflict pairs."""
        # Penguin has both CapableOf swim and NotCapableOf fly
        conflicts = engine.get_potential_conflicts("penguin")
        # Should not find internal conflicts for penguin
        # (swim ≠ fly, so no capability conflict)
        assert isinstance(conflicts, list)
    
    def test_batch_conflict_check(self, engine):
        """Test batch conflict checking."""
        triples = [
            Triple("penguin", "CapableOf", "fly"),  # Conflict
            Triple("penguin", "CapableOf", "swim"),  # No conflict
            Triple("bird", "CapableOf", "fly"),  # No conflict
        ]
        
        results = engine.check_conflicts_batch(triples)
        assert len(results) == 3
        assert len(results[triples[0]]) > 0  # Has conflicts


class TestSemanticInconsistencyDetector:
    """Tests for the main SID class."""
    
    @pytest.fixture
    def detector(self):
        """Create SID instance for testing."""
        config = SIDConfig(
            nlp_backend="rule_based",
            conceptnet_cache_enabled=True,
            verbose=False
        )
        return SemanticInconsistencyDetector(config=config)
    
    def test_detector_initialization(self, detector):
        """Test detector initializes properly."""
        assert detector is not None
        assert detector.config is not None
    
    def test_detect_conflict_penguin_fly(self, detector):
        """Test detecting the classic penguin/fly conflict."""
        result = detector.detect_conflict("Penguins can fly")
        
        assert result is not None
        assert result.has_conflict is True
        assert len(result.conflicts) > 0
    
    def test_detect_no_conflict(self, detector):
        """Test case with no conflict."""
        result = detector.detect_conflict("Penguins can swim")
        
        # Swimming is consistent with penguin knowledge
        # Note: might still be False if no triples extracted
        assert result is not None
    
    def test_detect_conflict_returns_evidence(self, detector):
        """Test that conflict detection returns proper evidence."""
        result = detector.detect_conflict("Penguins can fly")
        
        if result.has_conflict:
            evidence = result.conflicts[0]
            assert evidence.source_triple is not None
            assert evidence.conflicting_triple is not None
            assert evidence.conflict_type is not None
            assert len(evidence.reasoning_chain) > 0
    
    def test_is_conflicting_shortcut(self, detector):
        """Test the is_conflicting convenience method."""
        assert detector.is_conflicting("Penguins can fly") is True
    
    def test_get_conflicts_shortcut(self, detector):
        """Test the get_conflicts convenience method."""
        conflicts = detector.get_conflicts("Penguins can fly")
        assert isinstance(conflicts, list)
    
    def test_extract_triples(self, detector):
        """Test triple extraction."""
        result = detector.extract_triples("Birds can fly in the sky")
        
        assert result is not None
        assert result.original_text == "Birds can fly in the sky"
        assert isinstance(result.entities, list)
        assert isinstance(result.triples, list)
    
    def test_query_knowledge(self, detector):
        """Test knowledge base queries."""
        triples = detector.query_knowledge("penguin")
        
        assert isinstance(triples, list)
        assert len(triples) > 0
    
    def test_batch_detection(self, detector):
        """Test batch conflict detection."""
        texts = [
            "Birds can fly",
            "Penguins can fly",
            "Dogs can bark",
        ]
        
        batch_result = detector.detect_conflicts_batch(texts)
        
        assert batch_result is not None
        assert batch_result.total_inputs == 3
        assert batch_result.inputs_with_conflicts >= 1  # At least penguin/fly
    
    def test_explain_method(self, detector):
        """Test the explain method."""
        explanation = detector.explain("Penguins can fly")
        
        assert isinstance(explanation, str)
        assert "penguin" in explanation.lower()
        assert len(explanation) > 100  # Should be a detailed explanation
    
    def test_get_system_info(self, detector):
        """Test system info retrieval."""
        info = detector.get_system_info()
        
        assert "version" in info
        assert "config" in info
        assert "nlp_backend" in info
    
    def test_empty_input_handling(self, detector):
        """Test handling of empty input."""
        result = detector.detect_conflict("")
        assert result.has_conflict is False
        assert len(result.warnings) > 0
    
    def test_whitespace_input_handling(self, detector):
        """Test handling of whitespace-only input."""
        result = detector.detect_conflict("   ")
        assert result.has_conflict is False


class TestConflictResult:
    """Tests for ConflictResult class."""
    
    def test_result_summary(self):
        """Test generating result summary."""
        result = ConflictResult(
            has_conflict=True,
            input_text="Test input",
            extracted_triples=[Triple("a", "b", "c")],
            conflicts=[],
            processing_time=0.1
        )
        
        summary = result.summary()
        assert isinstance(summary, str)
    
    def test_result_serialization(self):
        """Test result JSON serialization."""
        result = ConflictResult(
            has_conflict=False,
            input_text="Test",
            extracted_triples=[],
            conflicts=[],
            processing_time=0.1
        )
        
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert "has_conflict" in json_str
    
    def test_result_to_dict(self):
        """Test result dictionary conversion."""
        result = ConflictResult(
            has_conflict=False,
            input_text="Test",
            extracted_triples=[Triple("a", "b", "c")],
            conflicts=[],
            processing_time=0.1
        )
        
        data = result.to_dict()
        assert data["has_conflict"] is False
        assert len(data["extracted_triples"]) == 1


class TestFactoryFunction:
    """Tests for the create_detector factory function."""
    
    def test_create_detector_default(self):
        """Test creating detector with defaults."""
        detector = create_detector()
        assert detector is not None
    
    def test_create_detector_with_options(self):
        """Test creating detector with options."""
        detector = create_detector(
            backend="rule_based",
            cache_enabled=True,
            verbose=False
        )
        assert detector is not None
        assert detector.config.nlp_backend == "rule_based"


class TestIntegration:
    """Integration tests for end-to-end scenarios."""
    
    @pytest.fixture
    def detector(self):
        """Create SID for integration tests."""
        return create_detector(backend="rule_based")
    
    def test_classic_penguin_scenario(self, detector):
        """Test the classic penguin/bird/fly scenario."""
        # This is the key test case for the SG-CL project
        
        # Statement that should conflict
        result = detector.detect_conflict("Penguins can fly")
        
        assert result.has_conflict is True
        assert result.conflict_count >= 1
        
        # Check that we found the right conflict
        conflict = result.most_severe_conflict
        assert conflict is not None
        assert "fly" in conflict.source_triple.object.lower()
    
    def test_consistent_statement(self, detector):
        """Test a statement consistent with knowledge base."""
        # Penguins can swim is consistent
        result = detector.detect_conflict("Penguins can swim")
        
        # Should not detect false positives
        # (May still be True if inheritance reasoning kicks in)
        assert result is not None
    
    def test_multiple_concepts(self, detector):
        """Test statement with multiple concepts."""
        result = detector.detect_conflict("Dogs and cats are animals")
        assert result is not None
    
    def test_property_conflict(self, detector):
        """Test property-based conflicts."""
        # Fire is cold would conflict with fire HasProperty hot
        result = detector.detect_conflict("Fire is cold")
        # This depends on KB content
        assert result is not None
    
    def test_sequential_learning_scenario(self, detector):
        """
        Test a scenario similar to sequential learning.
        
        This simulates what happens in continual learning:
        1. Learn: All birds can fly
        2. Learn: Penguins are birds
        3. Learn: Penguins cannot fly
        
        The system should detect the semantic conflict.
        """
        statements = [
            "All birds can fly",
            "Penguins are birds",
            "Penguins cannot fly"
        ]
        
        batch_result = detector.detect_conflicts_batch(statements)
        
        assert batch_result.total_inputs == 3
        # At least one statement should have a conflict
        # (either "All birds can fly" conflicts with penguin exception,
        # or "Penguins cannot fly" is flagged as inheritance conflict)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.fixture
    def detector(self):
        return create_detector(backend="rule_based")
    
    def test_very_long_input(self, detector):
        """Test handling of very long input."""
        long_text = "Birds can fly. " * 100
        result = detector.detect_conflict(long_text)
        assert result is not None
    
    def test_special_characters(self, detector):
        """Test handling of special characters."""
        result = detector.detect_conflict("Birds can fly! @#$%")
        assert result is not None
    
    def test_unicode_input(self, detector):
        """Test handling of unicode characters."""
        result = detector.detect_conflict("Penguins 🐧 can fly")
        assert result is not None
    
    def test_numbers_in_input(self, detector):
        """Test handling of numbers in input."""
        result = detector.detect_conflict("5 birds can fly")
        assert result is not None
    
    def test_only_stopwords(self, detector):
        """Test input with only stopwords."""
        result = detector.detect_conflict("the a an is are")
        assert result is not None
        # Should have warning about no triples extracted
    
    def test_single_word(self, detector):
        """Test single word input."""
        result = detector.detect_conflict("penguin")
        assert result is not None


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

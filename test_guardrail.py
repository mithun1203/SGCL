"""
Unit Tests for Symbolic Guardrail System.

Tests the guardrail generation and SID-gated control mechanisms.
"""

import unittest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from guardrail.guardrail_generator import GuardrailGenerator, GuardrailFact
from guardrail.guardrail_controller import GuardrailController, TrainingBatch


class TestGuardrailGenerator(unittest.TestCase):
    """Test guardrail fact generation."""
    
    def setUp(self):
        """Initialize generator with test KB."""
        kb_path = Path(__file__).parent / 'sid' / 'knowledge_base.json'
        self.generator = GuardrailGenerator(str(kb_path))
    
    def test_generate_creates_facts(self):
        """Test that generate() creates guardrail facts."""
        facts = self.generator.generate("penguin", "/r/CapableOf", "fly", max_facts=4)
        
        self.assertIsInstance(facts, list)
        self.assertGreater(len(facts), 0, "Should generate at least one fact")
        self.assertLessEqual(len(facts), 4, "Should not exceed max_facts")
        
        for fact in facts:
            self.assertIsInstance(fact, GuardrailFact)
            self.assertIsInstance(fact.sentence, str)
            self.assertGreater(len(fact.sentence), 0)
    
    def test_generate_respects_budget(self):
        """Test that guardrails respect the 2-4 fact budget."""
        for max_facts in [2, 3, 4]:
            facts = self.generator.generate("penguin", "/r/CapableOf", "fly", max_facts=max_facts)
            self.assertLessEqual(len(facts), max_facts, 
                               f"Should not exceed max_facts={max_facts}")
    
    def test_general_rule_generation(self):
        """Test general rule reinforcement."""
        facts = self.generator.generate("penguin", "/r/CapableOf", "fly", max_facts=4)
        
        # Should include parent class rule: "Birds can fly."
        general_rules = [f for f in facts if f.fact_type == 'general_rule']
        self.assertGreater(len(general_rules), 0, "Should generate general rule")
        
        rule = general_rules[0]
        self.assertIn("Birds", rule.sentence)
        self.assertIn("fly", rule.sentence)
    
    def test_sibling_examples(self):
        """Test sibling example generation."""
        facts = self.generator.generate("penguin", "/r/CapableOf", "fly", max_facts=4)
        
        # Should include sibling examples
        siblings = [f for f in facts if f.fact_type == 'sibling_example']
        self.assertGreater(len(siblings), 0, "Should generate sibling examples")
        
        # Check format
        for sibling in siblings:
            self.assertIn("can fly", sibling.sentence.lower())
    
    def test_hierarchy_preservation(self):
        """Test hierarchy fact generation."""
        facts = self.generator.generate("penguin", "/r/CapableOf", "fly", max_facts=4)
        
        # Should include hierarchy: "Penguins are birds."
        hierarchy = [f for f in facts if f.fact_type == 'hierarchy']
        self.assertGreater(len(hierarchy), 0, "Should generate hierarchy fact")
        
        h = hierarchy[0]
        self.assertIn("Penguins", h.sentence)
        self.assertIn("birds", h.sentence)
    
    def test_entity_normalization(self):
        """Test that plural entities are normalized to singular."""
        # Both should work
        facts_singular = self.generator.generate("penguin", "/r/CapableOf", "fly")
        facts_plural = self.generator.generate("penguins", "/r/CapableOf", "fly")
        
        self.assertEqual(len(facts_singular), len(facts_plural), 
                        "Plural and singular should produce same facts")
    
    def test_natural_language_output(self):
        """Test that all output is natural language (no symbolic notation)."""
        facts = self.generator.generate("penguin", "/r/CapableOf", "fly", max_facts=4)
        
        for fact in facts:
            # Should not contain symbolic notation like /r/ or /c/en/
            self.assertNotIn("/r/", fact.sentence, "Should not contain /r/ notation")
            self.assertNotIn("/c/en/", fact.sentence, "Should not contain /c/en/ notation")
            
            # Should be proper sentences (capitalized, ends with punctuation)
            self.assertTrue(fact.sentence[0].isupper(), "Should start with capital")
            self.assertTrue(fact.sentence.endswith('.'), "Should end with period")


class TestGuardrailController(unittest.TestCase):
    """Test SID-gated guardrail control."""
    
    def setUp(self):
        """Initialize controller with test KB."""
        kb_path = Path(__file__).parent / 'sid' / 'knowledge_base.json'
        self.controller = GuardrailController(max_guardrails=4)
        # Knowledge base for testing
        self.kb = [
            "Birds can fly.",
            "Penguins are birds.",
            "Penguins cannot fly."
        ]
    
    def test_no_conflict_no_guardrails(self):
        """Test that no guardrails are added when no conflict detected."""
        batch = ["Eagles have sharp talons.", "Birds can fly."]
        result = self.controller.process_batch(batch, self.kb)
        
        self.assertIsInstance(result, TrainingBatch)
        self.assertFalse(result.has_conflict, "Should not detect conflict")
        self.assertEqual(len(result.guardrail_samples), 0, "Should not add guardrails")
        self.assertEqual(result.original_samples, batch)
    
    def test_conflict_triggers_guardrails(self):
        """Test that guardrails are added when conflict detected."""
        batch = ["Penguins can fly."]
        result = self.controller.process_batch(batch, self.kb)
        
        self.assertTrue(result.has_conflict, "Should detect conflict")
        self.assertGreater(len(result.guardrail_samples), 0, 
                          "Should add guardrails for conflict")
        self.assertLessEqual(len(result.guardrail_samples), 4, 
                            "Should respect max_guardrails budget")
    
    def test_augmented_batch_includes_guardrails(self):
        """Test that augmented batch includes original + guardrails."""
        batch = ["Penguins can fly."]
        result = self.controller.process_batch(batch, self.kb)
        
        if result.has_conflict:
            all_samples = result.original_samples + result.guardrail_samples
            
            # Should have original
            self.assertIn(batch[0], all_samples)
            
            # Should have guardrails
            self.assertGreater(len(all_samples), len(batch), 
                              "Augmented batch should be larger")
    
    def test_hard_gating(self):
        """Test hard SID-gating (guardrails only when conflict detected)."""
        # No conflict batch
        batch_clean = ["Eagles have sharp talons."]
        result_clean = self.controller.process_batch(batch_clean, self.kb)
        
        # Conflict batch
        batch_conflict = ["Penguins can fly."]
        result_conflict = self.controller.process_batch(batch_conflict, self.kb)
        
        # Hard gating: clean gets no guardrails, conflict gets guardrails
        self.assertEqual(len(result_clean.guardrail_samples), 0, 
                        "Clean batch should have 0 guardrails")
        
        if result_conflict.has_conflict:
            self.assertGreater(len(result_conflict.guardrail_samples), 0, 
                             "Conflict batch should have guardrails")
    
    def test_guardrail_quality(self):
        """Test that generated guardrails are semantically relevant."""
        batch = ["Penguins can fly."]
        result = self.controller.process_batch(batch, self.kb)
        
        if result.has_conflict:
            guardrails = result.guardrail_samples
            
            # Should contain relevant concepts
            guardrail_text = " ".join(guardrails).lower()
            self.assertTrue(
                "bird" in guardrail_text or "penguin" in guardrail_text,
                "Guardrails should mention relevant concepts"
            )


class TestIntegration(unittest.TestCase):
    """Integration tests for complete guardrail system."""
    
    def setUp(self):
        """Initialize controller."""
        self.controller = GuardrailController(max_guardrails=4)
        self.kb = [
            "Birds can fly.",
            "Penguins are birds.",
            "Penguins cannot fly."
        ]
    
    def test_full_workflow(self):
        """Test complete workflow: detect → generate → augment."""
        # Scenario: conflicting knowledge
        batch = ["Penguins can fly."]
        
        # Process
        result = self.controller.process_batch(batch, self.kb)
        
        # Verify
        self.assertTrue(result.has_conflict)
        self.assertGreaterEqual(len(result.guardrail_samples), 2)  # At least 2 facts
        self.assertLessEqual(len(result.guardrail_samples), 4)     # At most 4 facts
        
        # Check guardrail content
        guardrails = result.guardrail_samples
        
        # Should have general rule
        has_general = any("Birds can fly" in g for g in guardrails)
        self.assertTrue(has_general, "Should include general rule")
        
        # Should have hierarchy
        has_hierarchy = any("Penguins are birds" in g for g in guardrails)
        self.assertTrue(has_hierarchy, "Should include hierarchy")
    
    def test_batch_processing_statistics(self):
        """Test statistics tracking across multiple batches."""
        batches = [
            ["Eagles have sharp talons."],
            ["Penguins can fly."],
            ["Birds have wings."],
        ]
        
        results = [self.controller.process_batch(b, self.kb) for b in batches]
        
        # At least one should have conflict (Penguins can fly)
        has_any_conflict = any(r.has_conflict for r in results)
        self.assertTrue(has_any_conflict, "Should detect at least one conflict")
        
        # Count guardrails
        total_guardrails = sum(len(r.guardrail_samples) for r in results)
        self.assertGreater(total_guardrails, 0, "Should generate guardrails")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)

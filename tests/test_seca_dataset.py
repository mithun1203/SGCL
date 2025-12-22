"""
Test Suite for SeCA Dataset Module
===================================

Comprehensive tests for the Semantic Consistency Aware Dataset.
Ensures publishable quality for the SeCA component of SG-CL.

Author: Mithun Naik
Project: SGCL Capstone
Run with: pytest tests/test_seca_dataset.py -v
"""

import pytest
import json
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sid.seca_dataset import (
    SeCADataset,
    Sample,
    Task,
    ConflictType,
    create_seca_dataset,
    print_dataset_summary
)


class TestConflictType:
    """Tests for ConflictType enum."""
    
    def test_conflict_types_exist(self):
        """Test that all expected conflict types are defined."""
        expected = ['NONE', 'INHERITANCE', 'PROPERTY', 'CAPABILITY', 
                   'NEGATION', 'TEMPORAL', 'EXCEPTION']
        for ct in expected:
            assert hasattr(ConflictType, ct), f"Missing ConflictType.{ct}"
    
    def test_conflict_type_values(self):
        """Test that conflict types have correct string values."""
        assert ConflictType.NONE.value == "none"
        assert ConflictType.INHERITANCE.value == "inheritance"
        assert ConflictType.PROPERTY.value == "property"
        assert ConflictType.CAPABILITY.value == "capability"
        assert ConflictType.NEGATION.value == "negation"


class TestSample:
    """Tests for the Sample dataclass."""
    
    def test_sample_creation_basic(self):
        """Test basic sample creation."""
        sample = Sample(text="Birds can fly.")
        assert sample.text == "Birds can fly."
        assert sample.sample_id == 0
        assert sample.has_conflict == False
        assert sample.conflict_type == ConflictType.NONE
        assert sample.conflict_with is None
    
    def test_sample_with_conflict(self):
        """Test sample creation with conflict."""
        sample = Sample(
            text="Penguins can fly.",
            sample_id=1,
            has_conflict=True,
            conflict_type=ConflictType.INHERITANCE,
            conflict_with="Penguins cannot fly",
            entities=["penguin", "fly"],
            relations=["CapableOf"]
        )
        assert sample.has_conflict == True
        assert sample.conflict_type == ConflictType.INHERITANCE
        assert sample.conflict_with == "Penguins cannot fly"
        assert "penguin" in sample.entities
        assert "CapableOf" in sample.relations
    
    def test_sample_to_dict(self):
        """Test sample serialization."""
        sample = Sample(
            text="Fire is cold.",
            sample_id=5,
            has_conflict=True,
            conflict_type=ConflictType.PROPERTY,
            conflict_with="Fire is hot"
        )
        data = sample.to_dict()
        
        assert data["text"] == "Fire is cold."
        assert data["sample_id"] == 5
        assert data["has_conflict"] == True
        assert data["conflict_type"] == "property"
        assert data["conflict_with"] == "Fire is hot"
    
    def test_sample_default_values(self):
        """Test that default values are properly initialized."""
        sample = Sample(text="Test")
        assert sample.entities == []
        assert sample.relations == []
        assert sample.metadata == {}


class TestTask:
    """Tests for the Task dataclass."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            task_id=0,
            name="Bird Knowledge",
            description="Facts about birds"
        )
        assert task.task_id == 0
        assert task.name == "Bird Knowledge"
        assert task.description == "Facts about birds"
        assert len(task) == 0
    
    def test_task_add_sample(self):
        """Test adding samples to a task."""
        task = Task(task_id=0, name="Test", description="Test task")
        
        sample = task.add_sample("Birds can fly.", entities=["bird", "fly"])
        
        assert len(task) == 1
        assert sample.text == "Birds can fly."
        assert sample.sample_id == 0
    
    def test_task_add_conflict_sample(self):
        """Test adding a sample with conflict."""
        task = Task(task_id=0, name="Test", description="Test task")
        
        sample = task.add_sample(
            "Penguins can fly.",
            has_conflict=True,
            conflict_type=ConflictType.INHERITANCE,
            conflict_with="Penguins cannot fly"
        )
        
        assert task.expected_conflicts == 1
        assert sample.has_conflict == True
    
    def test_task_iteration(self):
        """Test iterating over task samples."""
        task = Task(task_id=0, name="Test", description="")
        task.add_sample("Sample 1")
        task.add_sample("Sample 2")
        task.add_sample("Sample 3")
        
        texts = [s.text for s in task]
        assert texts == ["Sample 1", "Sample 2", "Sample 3"]
    
    def test_task_to_dict(self):
        """Test task serialization."""
        task = Task(
            task_id=1,
            name="Animal Facts",
            description="Facts about animals",
            domain="animals",
            difficulty="easy"
        )
        task.add_sample("Dogs can bark.")
        
        data = task.to_dict()
        
        assert data["task_id"] == 1
        assert data["name"] == "Animal Facts"
        assert data["domain"] == "animals"
        assert data["sample_count"] == 1
        assert len(data["samples"]) == 1


class TestSeCADataset:
    """Tests for the SeCADataset class."""
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        dataset = SeCADataset(name="Test Dataset", version="1.0")
        
        assert dataset.name == "Test Dataset"
        assert dataset.version == "1.0"
        assert len(dataset) == 0
    
    def test_dataset_add_task(self):
        """Test adding tasks to dataset."""
        dataset = SeCADataset()
        
        task = dataset.add_task(
            name="Bird Facts",
            description="Facts about birds",
            domain="animals"
        )
        
        assert len(dataset) == 1
        assert task.task_id == 0
        assert task.name == "Bird Facts"
    
    def test_dataset_iteration(self):
        """Test iterating over dataset tasks."""
        dataset = SeCADataset()
        dataset.add_task(name="Task 1")
        dataset.add_task(name="Task 2")
        dataset.add_task(name="Task 3")
        
        names = [t.name for t in dataset]
        assert names == ["Task 1", "Task 2", "Task 3"]
    
    def test_dataset_indexing(self):
        """Test indexing into dataset."""
        dataset = SeCADataset()
        dataset.add_task(name="First")
        dataset.add_task(name="Second")
        
        assert dataset[0].name == "First"
        assert dataset[1].name == "Second"
    
    def test_dataset_total_samples(self):
        """Test total samples property."""
        dataset = SeCADataset()
        
        task1 = dataset.add_task(name="Task 1")
        task1.add_sample("Sample 1")
        task1.add_sample("Sample 2")
        
        task2 = dataset.add_task(name="Task 2")
        task2.add_sample("Sample 3")
        
        assert dataset.total_samples == 3
    
    def test_dataset_total_conflicts(self):
        """Test total conflicts property."""
        dataset = SeCADataset()
        
        task1 = dataset.add_task(name="Task 1")
        task1.add_sample("Normal sample")
        task1.add_sample("Conflict sample", has_conflict=True,
                        conflict_type=ConflictType.CAPABILITY)
        
        task2 = dataset.add_task(name="Task 2")
        task2.add_sample("Another conflict", has_conflict=True,
                        conflict_type=ConflictType.PROPERTY)
        
        assert dataset.total_conflicts == 2
    
    def test_dataset_statistics(self):
        """Test getting dataset statistics."""
        dataset = SeCADataset(name="Stats Test", version="2.0")
        
        task1 = dataset.add_task(name="Task 1", domain="animals")
        task1.add_sample("Normal")
        task1.add_sample("Conflict", has_conflict=True,
                        conflict_type=ConflictType.INHERITANCE)
        
        stats = dataset.get_statistics()
        
        assert stats["name"] == "Stats Test"
        assert stats["version"] == "2.0"
        assert stats["total_tasks"] == 1
        assert stats["total_samples"] == 2
        assert stats["total_conflicts"] == 1
        assert stats["conflict_rate"] == 0.5
        assert "inheritance" in stats["conflict_types"]
        assert "animals" in stats["domains"]
    
    def test_dataset_save_and_load(self):
        """Test saving and loading dataset."""
        dataset = SeCADataset(name="Save Test")
        
        task = dataset.add_task(name="Test Task", domain="test")
        task.add_sample("Sample 1", entities=["entity1"])
        task.add_sample("Conflict", has_conflict=True,
                        conflict_type=ConflictType.NEGATION,
                        conflict_with="Contradicting fact")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                         delete=False) as f:
            temp_path = f.name
        
        try:
            dataset.save(temp_path)
            
            # Load it back
            loaded = SeCADataset.load(temp_path)
            
            assert loaded.name == "Save Test"
            assert len(loaded) == 1
            assert loaded[0].name == "Test Task"
            assert loaded.total_samples == 2
            assert loaded.total_conflicts == 1
            
            # Check conflict was preserved
            samples = list(loaded[0])
            conflict_sample = [s for s in samples if s.has_conflict][0]
            assert conflict_sample.conflict_type == ConflictType.NEGATION
            assert conflict_sample.conflict_with == "Contradicting fact"
        finally:
            os.unlink(temp_path)


class TestStandardDataset:
    """Tests for the standard pre-built dataset."""
    
    def test_create_standard_dataset(self):
        """Test creating the standard dataset."""
        dataset = SeCADataset.create_standard()
        
        assert dataset.name == "SeCA Standard Dataset"
        assert len(dataset) >= 5  # At least 5 tasks
        assert dataset.total_samples >= 20  # At least 20 samples
        assert dataset.total_conflicts >= 3  # At least 3 conflicts
    
    def test_standard_dataset_has_all_conflict_types(self):
        """Test that standard dataset covers multiple conflict types."""
        dataset = SeCADataset.create_standard()
        
        conflict_types = set()
        for task in dataset:
            for sample in task:
                if sample.has_conflict:
                    conflict_types.add(sample.conflict_type)
        
        # Should have at least these types
        assert ConflictType.INHERITANCE in conflict_types
        assert ConflictType.PROPERTY in conflict_types
        assert ConflictType.CAPABILITY in conflict_types
    
    def test_standard_dataset_domains(self):
        """Test that standard dataset has multiple domains."""
        dataset = SeCADataset.create_standard()
        
        domains = {task.domain for task in dataset}
        
        # Should have multiple domains
        assert len(domains) >= 2
        assert "animals" in domains
    
    def test_penguin_conflict_exists(self):
        """Test that the classic penguin-fly conflict is present."""
        dataset = SeCADataset.create_standard()
        
        penguin_conflict_found = False
        for task in dataset:
            for sample in task:
                if "penguin" in sample.text.lower() and "fly" in sample.text.lower():
                    if sample.has_conflict:
                        penguin_conflict_found = True
                        break
        
        assert penguin_conflict_found, "Penguin-fly conflict not found"


class TestMinimalDataset:
    """Tests for the minimal dataset."""
    
    def test_create_minimal_dataset(self):
        """Test creating the minimal dataset."""
        dataset = SeCADataset.create_minimal()
        
        assert dataset.name == "SeCA Minimal Dataset"
        assert len(dataset) >= 2  # At least 2 tasks
        assert dataset.total_conflicts >= 1  # At least 1 conflict


class TestCreateSeCADataset:
    """Tests for the create_seca_dataset factory function."""
    
    def test_create_standard(self):
        """Test creating standard dataset via factory."""
        dataset = create_seca_dataset("standard")
        assert "Standard" in dataset.name
    
    def test_create_minimal(self):
        """Test creating minimal dataset via factory."""
        dataset = create_seca_dataset("minimal")
        assert "Minimal" in dataset.name
    
    def test_invalid_variant(self):
        """Test that invalid variant raises error."""
        with pytest.raises(ValueError):
            create_seca_dataset("invalid_variant")


class TestDatasetIntegration:
    """Integration tests for SeCA dataset with SID pipeline."""
    
    def test_dataset_json_format(self):
        """Test that dataset JSON format is correct."""
        dataset = SeCADataset.create_minimal()
        data = dataset.to_dict()
        
        # Verify JSON structure
        assert "name" in data
        assert "version" in data
        assert "tasks" in data
        assert "statistics" in data
        
        # Verify task structure
        task = data["tasks"][0]
        assert "task_id" in task
        assert "name" in task
        assert "samples" in task
        
        # Verify sample structure
        sample = task["samples"][0]
        assert "text" in sample
        assert "has_conflict" in sample
        assert "conflict_type" in sample
    
    def test_dataset_roundtrip(self):
        """Test that dataset survives JSON roundtrip."""
        original = SeCADataset.create_standard()
        
        # Convert to JSON and back
        json_str = json.dumps(original.to_dict())
        data = json.loads(json_str)
        
        # Verify data integrity
        assert data["statistics"]["total_tasks"] == len(original)
        assert data["statistics"]["total_samples"] == original.total_samples
        assert data["statistics"]["total_conflicts"] == original.total_conflicts


class TestDatasetMetrics:
    """Tests for dataset metrics and statistics."""
    
    def test_conflict_rate_calculation(self):
        """Test that conflict rate is calculated correctly."""
        dataset = SeCADataset()
        
        task = dataset.add_task(name="Test")
        task.add_sample("Normal 1")
        task.add_sample("Normal 2")
        task.add_sample("Normal 3")
        task.add_sample("Conflict", has_conflict=True,
                        conflict_type=ConflictType.PROPERTY)
        
        stats = dataset.get_statistics()
        assert stats["conflict_rate"] == 0.25  # 1/4 = 25%
    
    def test_empty_dataset_stats(self):
        """Test statistics for empty dataset."""
        dataset = SeCADataset()
        stats = dataset.get_statistics()
        
        assert stats["total_tasks"] == 0
        assert stats["total_samples"] == 0
        assert stats["conflict_rate"] == 0.0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

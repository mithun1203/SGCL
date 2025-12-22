"""
SeCA Dataset - Semantic Consistency Aware Dataset
==================================================

A dataset specifically designed for evaluating Semantic Consistency in
Continual Learning scenarios. Contains sequential tasks that test:

1. Inheritance conflicts (penguin -> bird -> fly)
2. Property conflicts (fire is hot vs cold)
3. Capability exceptions (flightless birds)
4. Temporal knowledge (changing facts)
5. Negation handling

The dataset simulates real-world learning where knowledge arrives over
time and may contain potential contradictions.

Author: Mithun Naik
Project: SGCL Capstone
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime


class ConflictType(Enum):
    """Types of semantic conflicts in the dataset."""
    NONE = "none"
    INHERITANCE = "inheritance"       # penguin is bird, birds fly, but penguins don't
    PROPERTY = "property"             # fire is hot vs fire is cold
    CAPABILITY = "capability"         # can vs cannot do something
    NEGATION = "negation"             # direct negation
    TEMPORAL = "temporal"             # knowledge that changes over time
    EXCEPTION = "exception"           # exceptions to general rules


@dataclass
class Sample:
    """A single training sample."""
    text: str
    sample_id: int = 0
    has_conflict: bool = False
    conflict_type: ConflictType = ConflictType.NONE
    conflict_with: Optional[str] = None  # What it conflicts with
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "sample_id": self.sample_id,
            "has_conflict": self.has_conflict,
            "conflict_type": self.conflict_type.value,
            "conflict_with": self.conflict_with,
            "entities": self.entities,
            "relations": self.relations,
            "metadata": self.metadata
        }


@dataclass
class Task:
    """A task in continual learning - a group of related samples."""
    task_id: int
    name: str
    description: str
    samples: List[Sample] = field(default_factory=list)
    domain: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    expected_conflicts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)
    
    def add_sample(self, text: str, has_conflict: bool = False,
                   conflict_type: ConflictType = ConflictType.NONE,
                   conflict_with: Optional[str] = None,
                   entities: Optional[List[str]] = None,
                   relations: Optional[List[str]] = None) -> Sample:
        """Add a sample to this task."""
        sample = Sample(
            text=text,
            sample_id=len(self.samples),
            has_conflict=has_conflict,
            conflict_type=conflict_type,
            conflict_with=conflict_with,
            entities=entities or [],
            relations=relations or []
        )
        self.samples.append(sample)
        if has_conflict:
            self.expected_conflicts += 1
        return sample
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "expected_conflicts": self.expected_conflicts,
            "sample_count": len(self.samples),
            "samples": [s.to_dict() for s in self.samples],
            "metadata": self.metadata
        }


class SeCADataset:
    """
    Semantic Consistency Aware Dataset for Continual Learning.
    
    Provides sequential tasks designed to test semantic consistency
    in LLM fine-tuning scenarios.
    
    Features:
        - Pre-built task sequences with known conflicts
        - Customizable difficulty levels
        - Multiple domains (animals, science, common sense)
        - Ground truth conflict annotations
        - Compatible with SG-CL pipeline
    
    Example:
        >>> dataset = SeCADataset.create_standard()
        >>> for task in dataset:
        ...     print(f"Task {task.task_id}: {task.name}")
        ...     for sample in task:
        ...         if sample.has_conflict:
        ...             print(f"  [CONFLICT] {sample.text}")
    """
    
    def __init__(self, name: str = "SeCA Dataset", version: str = "1.0"):
        self.name = name
        self.version = version
        self.tasks: List[Task] = []
        self.created_at = datetime.now().isoformat()
        self._sample_counter = 0
        self._task_counter = 0
    
    def add_task(self, name: str, description: str = "",
                 domain: str = "general", 
                 difficulty: str = "medium") -> Task:
        """Create and add a new task."""
        task = Task(
            task_id=self._task_counter,
            name=name,
            description=description,
            domain=domain,
            difficulty=difficulty
        )
        self.tasks.append(task)
        self._task_counter += 1
        return task
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)
    
    def __getitem__(self, idx: int) -> Task:
        return self.tasks[idx]
    
    @property
    def total_samples(self) -> int:
        """Total number of samples across all tasks."""
        return sum(len(task) for task in self.tasks)
    
    @property
    def total_conflicts(self) -> int:
        """Total expected conflicts."""
        return sum(task.expected_conflicts for task in self.tasks)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        conflict_types = {}
        domains = {}
        
        for task in self.tasks:
            domains[task.domain] = domains.get(task.domain, 0) + 1
            for sample in task:
                if sample.has_conflict:
                    ct = sample.conflict_type.value
                    conflict_types[ct] = conflict_types.get(ct, 0) + 1
        
        return {
            "name": self.name,
            "version": self.version,
            "total_tasks": len(self.tasks),
            "total_samples": self.total_samples,
            "total_conflicts": self.total_conflicts,
            "conflict_rate": self.total_conflicts / max(self.total_samples, 1),
            "conflict_types": conflict_types,
            "domains": domains,
            "created_at": self.created_at
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "statistics": self.get_statistics(),
            "tasks": [task.to_dict() for task in self.tasks]
        }
    
    def save(self, path: str) -> None:
        """Save dataset to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'SeCADataset':
        """Load dataset from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset = cls(name=data.get("name", "SeCA Dataset"),
                      version=data.get("version", "1.0"))
        
        for task_data in data.get("tasks", []):
            task = dataset.add_task(
                name=task_data["name"],
                description=task_data.get("description", ""),
                domain=task_data.get("domain", "general"),
                difficulty=task_data.get("difficulty", "medium")
            )
            
            for sample_data in task_data.get("samples", []):
                task.add_sample(
                    text=sample_data["text"],
                    has_conflict=sample_data.get("has_conflict", False),
                    conflict_type=ConflictType(sample_data.get("conflict_type", "none")),
                    conflict_with=sample_data.get("conflict_with"),
                    entities=sample_data.get("entities", []),
                    relations=sample_data.get("relations", [])
                )
        
        return dataset
    
    # =========================================================================
    # Pre-built Dataset Creators
    # =========================================================================
    
    @classmethod
    def create_standard(cls) -> 'SeCADataset':
        """
        Create the standard SeCA dataset with all conflict types.
        
        Returns:
            SeCADataset with multiple tasks testing different conflict types
        """
        dataset = cls(name="SeCA Standard Dataset", version="1.0")
        
        # =====================================================================
        # Task 1: General Bird Knowledge (No conflicts)
        # =====================================================================
        task1 = dataset.add_task(
            name="General Bird Knowledge",
            description="Establish baseline bird knowledge",
            domain="animals",
            difficulty="easy"
        )
        task1.add_sample(
            "Birds are animals that can fly.",
            entities=["bird", "animal"],
            relations=["IsA", "CapableOf"]
        )
        task1.add_sample(
            "Birds have wings and feathers.",
            entities=["bird", "wing", "feather"],
            relations=["HasA"]
        )
        task1.add_sample(
            "Sparrows are small birds that can fly.",
            entities=["sparrow", "bird"],
            relations=["IsA", "CapableOf"]
        )
        task1.add_sample(
            "Eagles are large birds that can fly very high.",
            entities=["eagle", "bird"],
            relations=["IsA", "CapableOf"]
        )
        task1.add_sample(
            "Most birds build nests to lay their eggs.",
            entities=["bird", "nest", "egg"],
            relations=["CapableOf", "UsedFor"]
        )
        
        # =====================================================================
        # Task 2: Penguin Knowledge (Contains inheritance conflict!)
        # =====================================================================
        task2 = dataset.add_task(
            name="Penguin Knowledge",
            description="Penguin facts with flight conflict",
            domain="animals",
            difficulty="medium"
        )
        task2.add_sample(
            "Penguins are birds.",
            entities=["penguin", "bird"],
            relations=["IsA"]
        )
        task2.add_sample(
            "Penguins can fly.",  # CONFLICT!
            has_conflict=True,
            conflict_type=ConflictType.INHERITANCE,
            conflict_with="Penguins cannot fly (they are flightless)",
            entities=["penguin", "fly"],
            relations=["CapableOf"]
        )
        task2.add_sample(
            "Penguins are excellent swimmers.",
            entities=["penguin", "swim"],
            relations=["CapableOf"]
        )
        task2.add_sample(
            "Penguins live in Antarctica.",
            entities=["penguin", "Antarctica"],
            relations=["AtLocation"]
        )
        task2.add_sample(
            "Penguins eat fish and krill.",
            entities=["penguin", "fish", "krill"],
            relations=["UsedFor"]
        )
        
        # =====================================================================
        # Task 3: More Flightless Birds
        # =====================================================================
        task3 = dataset.add_task(
            name="Flightless Bird Knowledge",
            description="Other flightless birds",
            domain="animals",
            difficulty="medium"
        )
        task3.add_sample(
            "Ostriches are the largest birds in the world.",
            entities=["ostrich", "bird"],
            relations=["IsA", "HasProperty"]
        )
        task3.add_sample(
            "Ostriches cannot fly but can run very fast.",
            entities=["ostrich", "fly", "run"],
            relations=["NotCapableOf", "CapableOf"]
        )
        task3.add_sample(
            "Emus are large flightless birds from Australia.",
            entities=["emu", "bird", "Australia"],
            relations=["IsA", "AtLocation"]
        )
        task3.add_sample(
            "Kiwis are small flightless birds native to New Zealand.",
            entities=["kiwi", "bird", "New Zealand"],
            relations=["IsA", "AtLocation"]
        )
        
        # =====================================================================
        # Task 4: Fish and Aquatic Animals
        # =====================================================================
        task4 = dataset.add_task(
            name="Aquatic Animal Knowledge",
            description="Facts about fish and aquatic animals",
            domain="animals",
            difficulty="easy"
        )
        task4.add_sample(
            "Fish can swim in water.",
            entities=["fish", "swim", "water"],
            relations=["CapableOf", "AtLocation"]
        )
        task4.add_sample(
            "Fish breathe through gills.",
            entities=["fish", "gill"],
            relations=["HasA", "UsedFor"]
        )
        task4.add_sample(
            "Sharks are fish that can be dangerous.",
            entities=["shark", "fish"],
            relations=["IsA", "HasProperty"]
        )
        task4.add_sample(
            "Fish can walk on land.",  # CONFLICT!
            has_conflict=True,
            conflict_type=ConflictType.CAPABILITY,
            conflict_with="Fish cannot walk on land",
            entities=["fish", "walk", "land"],
            relations=["CapableOf"]
        )
        task4.add_sample(
            "Whales are mammals, not fish.",
            entities=["whale", "mammal", "fish"],
            relations=["IsA", "DistinctFrom"]
        )
        
        # =====================================================================
        # Task 5: Mammals
        # =====================================================================
        task5 = dataset.add_task(
            name="Mammal Knowledge",
            description="Facts about mammals",
            domain="animals",
            difficulty="easy"
        )
        task5.add_sample(
            "Dogs are mammals that can bark.",
            entities=["dog", "mammal", "bark"],
            relations=["IsA", "CapableOf"]
        )
        task5.add_sample(
            "Cats are mammals that can meow.",
            entities=["cat", "mammal", "meow"],
            relations=["IsA", "CapableOf"]
        )
        task5.add_sample(
            "Humans are mammals that can think and reason.",
            entities=["human", "mammal", "think"],
            relations=["IsA", "CapableOf"]
        )
        task5.add_sample(
            "Bats are the only mammals that can fly.",
            entities=["bat", "mammal", "fly"],
            relations=["IsA", "CapableOf"]
        )
        task5.add_sample(
            "Elephants are the largest land mammals.",
            entities=["elephant", "mammal"],
            relations=["IsA", "HasProperty"]
        )
        
        # =====================================================================
        # Task 6: Element Properties (Contains property conflict!)
        # =====================================================================
        task6 = dataset.add_task(
            name="Element Properties",
            description="Properties of natural elements",
            domain="science",
            difficulty="easy"
        )
        task6.add_sample(
            "Fire is hot and produces light.",
            entities=["fire", "hot", "light"],
            relations=["HasProperty", "Causes"]
        )
        task6.add_sample(
            "Water is wet and can be liquid, solid, or gas.",
            entities=["water", "wet", "liquid"],
            relations=["HasProperty"]
        )
        task6.add_sample(
            "Ice is frozen water and is cold.",
            entities=["ice", "water", "cold"],
            relations=["MadeOf", "HasProperty"]
        )
        task6.add_sample(
            "Fire is cold.",  # CONFLICT!
            has_conflict=True,
            conflict_type=ConflictType.PROPERTY,
            conflict_with="Fire is hot, not cold",
            entities=["fire", "cold"],
            relations=["HasProperty"]
        )
        task6.add_sample(
            "The sun is extremely hot.",
            entities=["sun", "hot"],
            relations=["HasProperty"]
        )
        
        # =====================================================================
        # Task 7: Human Capabilities
        # =====================================================================
        task7 = dataset.add_task(
            name="Human Capabilities",
            description="What humans can and cannot do",
            domain="common_sense",
            difficulty="medium"
        )
        task7.add_sample(
            "Humans can walk on two legs.",
            entities=["human", "walk"],
            relations=["CapableOf"]
        )
        task7.add_sample(
            "Humans can breathe underwater without equipment.",  # CONFLICT!
            has_conflict=True,
            conflict_type=ConflictType.CAPABILITY,
            conflict_with="Humans cannot breathe underwater",
            entities=["human", "breathe", "underwater"],
            relations=["CapableOf"]
        )
        task7.add_sample(
            "Humans can fly without any assistance.",  # CONFLICT!
            has_conflict=True,
            conflict_type=ConflictType.CAPABILITY,
            conflict_with="Humans cannot fly without machines",
            entities=["human", "fly"],
            relations=["CapableOf"]
        )
        task7.add_sample(
            "Humans can communicate using language.",
            entities=["human", "communicate", "language"],
            relations=["CapableOf", "UsedFor"]
        )
        task7.add_sample(
            "Humans need food and water to survive.",
            entities=["human", "food", "water"],
            relations=["HasPrerequisite"]
        )
        
        # =====================================================================
        # Task 8: Vehicle Knowledge
        # =====================================================================
        task8 = dataset.add_task(
            name="Vehicle Knowledge",
            description="Facts about vehicles",
            domain="objects",
            difficulty="easy"
        )
        task8.add_sample(
            "Cars have four wheels and can drive on roads.",
            entities=["car", "wheel", "road"],
            relations=["HasA", "CapableOf", "AtLocation"]
        )
        task8.add_sample(
            "Airplanes can fly through the sky.",
            entities=["airplane", "fly", "sky"],
            relations=["CapableOf", "AtLocation"]
        )
        task8.add_sample(
            "Boats can travel on water.",
            entities=["boat", "water"],
            relations=["CapableOf", "AtLocation"]
        )
        task8.add_sample(
            "Bicycles have two wheels and are powered by humans.",
            entities=["bicycle", "wheel", "human"],
            relations=["HasA", "UsedFor"]
        )
        task8.add_sample(
            "Submarines can travel underwater.",
            entities=["submarine", "underwater"],
            relations=["CapableOf", "AtLocation"]
        )
        
        # =====================================================================
        # Task 9: Negation Handling
        # =====================================================================
        task9 = dataset.add_task(
            name="Negation Knowledge",
            description="Testing negation handling",
            domain="common_sense",
            difficulty="hard"
        )
        task9.add_sample(
            "Snakes do not have legs.",
            entities=["snake", "leg"],
            relations=["NotHasA"]
        )
        task9.add_sample(
            "Snakes have legs.",  # CONFLICT!
            has_conflict=True,
            conflict_type=ConflictType.NEGATION,
            conflict_with="Snakes do not have legs",
            entities=["snake", "leg"],
            relations=["HasA"]
        )
        task9.add_sample(
            "Rocks are not alive.",
            entities=["rock", "alive"],
            relations=["NotHasProperty"]
        )
        task9.add_sample(
            "Plants can produce their own food through photosynthesis.",
            entities=["plant", "food", "photosynthesis"],
            relations=["CapableOf"]
        )
        task9.add_sample(
            "Animals cannot photosynthesize.",
            entities=["animal", "photosynthesize"],
            relations=["NotCapableOf"]
        )
        
        # =====================================================================
        # Task 10: Mixed Review (Multiple potential conflicts)
        # =====================================================================
        task10 = dataset.add_task(
            name="Mixed Knowledge Review",
            description="Mixed facts to test overall consistency",
            domain="general",
            difficulty="hard"
        )
        task10.add_sample(
            "All mammals give birth to live young.",
            entities=["mammal", "birth"],
            relations=["CapableOf"]
        )
        task10.add_sample(
            "The platypus is a mammal that lays eggs.",  # Exception!
            entities=["platypus", "mammal", "egg"],
            relations=["IsA", "CapableOf"]
        )
        task10.add_sample(
            "Water boils at 100 degrees Celsius at sea level.",
            entities=["water", "boil", "temperature"],
            relations=["HasProperty"]
        )
        task10.add_sample(
            "The moon produces its own light.",  # CONFLICT!
            has_conflict=True,
            conflict_type=ConflictType.PROPERTY,
            conflict_with="The moon reflects sunlight, it doesn't produce light",
            entities=["moon", "light"],
            relations=["Causes"]
        )
        task10.add_sample(
            "Spiders have eight legs.",
            entities=["spider", "leg"],
            relations=["HasA"]
        )
        
        return dataset
    
    @classmethod
    def create_minimal(cls) -> 'SeCADataset':
        """Create a minimal dataset for quick testing."""
        dataset = cls(name="SeCA Minimal Dataset", version="1.0")
        
        # Task 1: Basic bird facts
        task1 = dataset.add_task(
            name="Bird Basics",
            description="Basic bird knowledge",
            domain="animals",
            difficulty="easy"
        )
        task1.add_sample("Birds can fly.", entities=["bird", "fly"])
        task1.add_sample("Birds have wings.", entities=["bird", "wing"])
        
        # Task 2: Penguin conflict
        task2 = dataset.add_task(
            name="Penguin Facts",
            description="Penguin knowledge with conflict",
            domain="animals",
            difficulty="medium"
        )
        task2.add_sample("Penguins are birds.", entities=["penguin", "bird"])
        task2.add_sample(
            "Penguins can fly.",
            has_conflict=True,
            conflict_type=ConflictType.INHERITANCE,
            conflict_with="Penguins cannot fly",
            entities=["penguin", "fly"]
        )
        
        # Task 3: Property conflict
        task3 = dataset.add_task(
            name="Properties",
            description="Property knowledge with conflict",
            domain="science",
            difficulty="easy"
        )
        task3.add_sample("Fire is hot.", entities=["fire", "hot"])
        task3.add_sample(
            "Fire is cold.",
            has_conflict=True,
            conflict_type=ConflictType.PROPERTY,
            conflict_with="Fire is hot, not cold",
            entities=["fire", "cold"]
        )
        
        return dataset
    
    @classmethod
    def create_animal_domain(cls) -> 'SeCADataset':
        """Create a dataset focused on animal knowledge."""
        dataset = cls(name="SeCA Animal Domain", version="1.0")
        
        # Many animal-related tasks with increasing complexity
        # ... (implementation similar to standard)
        
        return cls.create_standard()  # Use standard for now
    
    @classmethod  
    def create_from_pipeline(cls, pipeline_dataset) -> 'SeCADataset':
        """
        Create SeCA dataset from existing pipeline dataset.
        
        Useful for converting simple pipeline tasks to full SeCA format.
        """
        dataset = cls(name="Converted SeCA Dataset", version="1.0")
        
        for task in pipeline_dataset:
            seca_task = dataset.add_task(
                name=task.name,
                description=f"Converted from task {task.task_id}"
            )
            for sample_text in task.samples:
                seca_task.add_sample(sample_text)
        
        return dataset


# =============================================================================
# Utility Functions
# =============================================================================

def create_seca_dataset(variant: str = "standard") -> SeCADataset:
    """
    Create a SeCA dataset by name.
    
    Args:
        variant: One of "standard", "minimal", "animal"
    
    Returns:
        SeCADataset instance
    """
    creators = {
        "standard": SeCADataset.create_standard,
        "minimal": SeCADataset.create_minimal,
        "animal": SeCADataset.create_animal_domain,
    }
    
    creator = creators.get(variant.lower())
    if creator is None:
        raise ValueError(f"Unknown variant: {variant}. "
                        f"Choose from: {list(creators.keys())}")
    
    return creator()


def print_dataset_summary(dataset: SeCADataset) -> None:
    """Print a summary of the dataset."""
    stats = dataset.get_statistics()
    
    print("=" * 60)
    print(f"  {stats['name']} (v{stats['version']})")
    print("=" * 60)
    print()
    print(f"  Total Tasks: {stats['total_tasks']}")
    print(f"  Total Samples: {stats['total_samples']}")
    print(f"  Total Conflicts: {stats['total_conflicts']}")
    print(f"  Conflict Rate: {stats['conflict_rate']:.1%}")
    print()
    
    print("  Conflict Types:")
    for ctype, count in stats['conflict_types'].items():
        print(f"    - {ctype}: {count}")
    print()
    
    print("  Domains:")
    for domain, count in stats['domains'].items():
        print(f"    - {domain}: {count} tasks")
    print()
    
    print("  Tasks:")
    for task in dataset:
        conflicts = f"[{task.expected_conflicts} conflicts]" if task.expected_conflicts else ""
        print(f"    {task.task_id}. {task.name} ({len(task)} samples) {conflicts}")
    print()


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("Creating SeCA Standard Dataset...")
    print()
    
    # Create standard dataset
    dataset = create_seca_dataset("standard")
    
    # Print summary
    print_dataset_summary(dataset)
    
    # Show some samples with conflicts
    print("=" * 60)
    print("  SAMPLES WITH CONFLICTS")
    print("=" * 60)
    print()
    
    for task in dataset:
        for sample in task:
            if sample.has_conflict:
                print(f"  Task: {task.name}")
                print(f"  Sample: \"{sample.text}\"")
                print(f"  Type: {sample.conflict_type.value}")
                print(f"  Conflicts with: {sample.conflict_with}")
                print()
    
    # Save to file
    output_path = Path(__file__).parent / "seca_dataset.json"
    dataset.save(str(output_path))
    print(f"Dataset saved to: {output_path}")

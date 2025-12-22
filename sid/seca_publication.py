"""
SeCA Dataset - Publication Version
===================================

Dataset for "Semantic Consistency Aware Continual Learning"
Designed for academic publication with proper evaluation splits.

Dataset Size: 320 samples across 8 tasks (40 samples each)
- Non-conflict: 140 samples
- True conflict: 140 samples  
- Hard/ambiguous: 40 samples

Author: Mithun Naik
Project: SGCL Capstone
Version: 2.0 (Publication Ready)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime


class ConflictLabel(Enum):
    """Classification labels for samples."""
    NO_CONFLICT = "no_conflict"
    CONFLICT = "conflict"
    AMBIGUOUS = "ambiguous"


class ConflictCategory(Enum):
    """Types of conflicts in the dataset."""
    NONE = "none"
    EXCEPTION_VIOLATION = "exception_violation"
    DIRECT_CONTRADICTION = "direct_contradiction"
    PARAPHRASE_CONFLICT = "paraphrase_conflict"
    MULTIHOP_REASONING = "multihop_reasoning"
    DELAYED_CONFLICT = "delayed_conflict"
    ATTRIBUTE_CONFLICT = "attribute_conflict"


@dataclass
class SeCAAnnotation:
    """
    Full annotation for a sample (publication format).
    
    Required fields for publication:
    - task_id: Task identifier
    - sentence: The statement
    - label: conflict/no_conflict/ambiguous
    - conflicts_with: List of conflicting statements (if any)
    - conflict_type: Category of conflict
    """
    task_id: int
    sample_id: int
    sentence: str
    label: ConflictLabel
    conflicts_with: List[str] = field(default_factory=list)
    conflict_type: ConflictCategory = ConflictCategory.NONE
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "sentence": self.sentence,
            "label": self.label.value,
            "conflicts_with": self.conflicts_with,
            "conflict_type": self.conflict_type.value,
            "entities": self.entities,
            "relations": self.relations,
            "reasoning_chain": self.reasoning_chain,
            "difficulty": self.difficulty
        }


@dataclass
class SeCATaskV2:
    """Task in the publication dataset."""
    task_id: int
    name: str
    description: str
    samples: List[SeCAAnnotation] = field(default_factory=list)
    
    def add_sample(
        self,
        sentence: str,
        label: ConflictLabel = ConflictLabel.NO_CONFLICT,
        conflicts_with: Optional[List[str]] = None,
        conflict_type: ConflictCategory = ConflictCategory.NONE,
        entities: Optional[List[str]] = None,
        relations: Optional[List[str]] = None,
        reasoning_chain: Optional[List[str]] = None,
        difficulty: str = "medium"
    ) -> SeCAAnnotation:
        """Add a sample to the task."""
        annotation = SeCAAnnotation(
            task_id=self.task_id,
            sample_id=len(self.samples),
            sentence=sentence,
            label=label,
            conflicts_with=conflicts_with or [],
            conflict_type=conflict_type,
            entities=entities or [],
            relations=relations or [],
            reasoning_chain=reasoning_chain or [],
            difficulty=difficulty
        )
        self.samples.append(annotation)
        return annotation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "sample_count": len(self.samples),
            "samples": [s.to_dict() for s in self.samples]
        }


class SeCAPublicationDataset:
    """
    Publication-ready SeCA Dataset.
    
    320 samples across 8 tasks (40 samples each).
    Designed for academic publication with proper splits.
    """
    
    def __init__(self, name: str = "SeCA Publication Dataset", version: str = "2.0"):
        self.name = name
        self.version = version
        self.tasks: List[SeCATaskV2] = []
        self.created_at = datetime.now().isoformat()
    
    def add_task(self, name: str, description: str) -> SeCATaskV2:
        """Add a new task."""
        task = SeCATaskV2(
            task_id=len(self.tasks) + 1,
            name=name,
            description=description
        )
        self.tasks.append(task)
        return task
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        total_samples = sum(len(task.samples) for task in self.tasks)
        
        label_counts = {
            "no_conflict": 0,
            "conflict": 0,
            "ambiguous": 0
        }
        
        conflict_type_counts = {}
        
        for task in self.tasks:
            for sample in task.samples:
                label_counts[sample.label.value] += 1
                if sample.conflict_type != ConflictCategory.NONE:
                    ct = sample.conflict_type.value
                    conflict_type_counts[ct] = conflict_type_counts.get(ct, 0) + 1
        
        return {
            "name": self.name,
            "version": self.version,
            "total_tasks": len(self.tasks),
            "total_samples": total_samples,
            "samples_per_task": total_samples // len(self.tasks) if self.tasks else 0,
            "label_distribution": label_counts,
            "conflict_types": conflict_type_counts,
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
        """Save dataset to JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'SeCAPublicationDataset':
        """Load dataset from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset = cls(name=data.get("name", "SeCA Dataset"),
                      version=data.get("version", "2.0"))
        
        for task_data in data.get("tasks", []):
            task = dataset.add_task(
                name=task_data["name"],
                description=task_data["description"]
            )
            
            for sample_data in task_data.get("samples", []):
                task.add_sample(
                    sentence=sample_data["sentence"],
                    label=ConflictLabel(sample_data["label"]),
                    conflicts_with=sample_data.get("conflicts_with", []),
                    conflict_type=ConflictCategory(sample_data.get("conflict_type", "none")),
                    entities=sample_data.get("entities", []),
                    relations=sample_data.get("relations", []),
                    reasoning_chain=sample_data.get("reasoning_chain", []),
                    difficulty=sample_data.get("difficulty", "medium")
                )
        
        return dataset
    
    @classmethod
    def create_publication_dataset(cls) -> 'SeCAPublicationDataset':
        """
        Create the full 320-sample publication dataset.
        
        8 Tasks × 40 Samples = 320 Total
        - 140 non-conflict
        - 140 true conflict
        - 40 hard/ambiguous
        """
        dataset = cls(name="SeCA Publication Dataset", version="2.0")
        
        # =====================================================================
        # TASK 1: General Rules (Base Semantics) - 40 samples
        # =====================================================================
        t1 = dataset.add_task(
            name="T1: General Rules (Base Semantics)",
            description="Universal rules establishing baseline knowledge"
        )
        
        # Birds (10 samples)
        base_facts_birds = [
            "Birds can fly.",
            "Birds have wings.",
            "Birds have feathers.",
            "Birds lay eggs.",
            "Birds build nests.",
            "Birds have beaks.",
            "Most birds can sing.",
            "Birds are warm-blooded animals.",
            "Birds have hollow bones.",
            "Birds have excellent vision."
        ]
        
        # Fish (10 samples)
        base_facts_fish = [
            "Fish live in water.",
            "Fish can swim.",
            "Fish breathe through gills.",
            "Fish have scales.",
            "Fish have fins.",
            "Fish are cold-blooded.",
            "Fish lay eggs in water.",
            "Fish can sense vibrations in water.",
            "Most fish cannot survive on land.",
            "Fish need oxygen from water."
        ]
        
        # Vehicles (10 samples)
        base_facts_vehicles = [
            "Vehicles need energy to move.",
            "Cars have four wheels.",
            "Cars run on fuel or electricity.",
            "Airplanes can fly.",
            "Boats travel on water.",
            "Bicycles are powered by humans.",
            "Motorcycles have two wheels.",
            "Trains run on tracks.",
            "Buses carry multiple passengers.",
            "Vehicles require maintenance."
        ]
        
        # Mammals (10 samples)
        base_facts_mammals = [
            "Mammals are warm-blooded.",
            "Mammals give birth to live young.",
            "Mammals have fur or hair.",
            "Mammals breathe air through lungs.",
            "Mammals produce milk for their young.",
            "Dogs are mammals that can bark.",
            "Cats are mammals that can meow.",
            "Humans are mammals that can speak.",
            "Elephants are large mammals.",
            "Whales are aquatic mammals."
        ]
        
        for fact in base_facts_birds + base_facts_fish + base_facts_vehicles + base_facts_mammals:
            t1.add_sample(
                sentence=fact,
                label=ConflictLabel.NO_CONFLICT,
                difficulty="easy"
            )
        
        # =====================================================================
        # TASK 2: Hierarchy / Taxonomy - 40 samples
        # =====================================================================
        t2 = dataset.add_task(
            name="T2: Hierarchy / Taxonomy",
            description="is-a and part-of relations"
        )
        
        taxonomy_facts = [
            # Birds (10)
            "Penguins are birds.",
            "Ostriches are birds.",
            "Eagles are birds.",
            "Sparrows are birds.",
            "Ducks are birds.",
            "Owls are birds.",
            "Parrots are birds.",
            "Flamingos are birds.",
            "Hummingbirds are birds.",
            "Ravens are birds.",
            
            # Fish (10)
            "Sharks are fish.",
            "Salmon are fish.",
            "Tuna are fish.",
            "Goldfish are fish.",
            "Eels are fish.",
            "Piranhas are fish.",
            "Clownfish are fish.",
            "Barracudas are fish.",
            "Stingrays are fish.",
            "Seahorses are fish.",
            
            # Vehicles (10)
            "Electric cars are vehicles.",
            "Hybrid cars are vehicles.",
            "Diesel trucks are vehicles.",
            "Sports cars are vehicles.",
            "Sedans are vehicles.",
            "SUVs are vehicles.",
            "Vans are vehicles.",
            "Jeeps are vehicles.",
            "Limousines are vehicles.",
            "Taxis are vehicles.",
            
            # Animals (10)
            "Dogs are animals.",
            "Cats are animals.",
            "Lions are predators.",
            "Rabbits are herbivores.",
            "Snakes are reptiles.",
            "Frogs are amphibians.",
            "Spiders are arachnids.",
            "Butterflies are insects.",
            "Whales are cetaceans.",
            "Bats are flying mammals."
        ]
        
        for fact in taxonomy_facts:
            t2.add_sample(
                sentence=fact,
                label=ConflictLabel.NO_CONFLICT,
                relations=["IsA"],
                difficulty="easy"
            )
        
        # =====================================================================
        # TASK 3: Attribute Inheritance - 40 samples
        # =====================================================================
        t3 = dataset.add_task(
            name="T3: Attribute Inheritance",
            description="Properties inherited from parent class"
        )
        
        inheritance_facts = [
            # Penguins inherit from birds (10)
            "Penguins have wings.",
            "Penguins have feathers.",
            "Penguins lay eggs.",
            "Penguins have beaks.",
            "Penguins are warm-blooded.",
            "Ostriches have wings.",
            "Ostriches have feathers.",
            "Ostriches lay eggs.",
            "Ostriches have beaks.",
            "Ostriches are warm-blooded.",
            
            # Fish attributes (10)
            "Sharks have gills.",
            "Sharks have fins.",
            "Sharks are cold-blooded.",
            "Salmon have gills.",
            "Salmon have fins.",
            "Salmon have scales.",
            "Tuna have gills.",
            "Tuna live in water.",
            "Goldfish have gills.",
            "Goldfish live in water.",
            
            # Electric car attributes (10)
            "Electric cars have four wheels.",
            "Electric cars need energy.",
            "Electric cars require maintenance.",
            "Electric cars can transport passengers.",
            "Electric cars have brakes.",
            "Hybrid cars have four wheels.",
            "Hybrid cars need energy.",
            "Diesel trucks have wheels.",
            "Sports cars have wheels.",
            "SUVs can transport passengers.",
            
            # Mammal attributes (10)
            "Dogs are warm-blooded.",
            "Dogs breathe air.",
            "Cats are warm-blooded.",
            "Cats have fur.",
            "Lions are warm-blooded.",
            "Lions breathe air.",
            "Elephants are warm-blooded.",
            "Elephants breathe air.",
            "Whales are warm-blooded.",
            "Bats are warm-blooded."
        ]
        
        for fact in inheritance_facts:
            t3.add_sample(
                sentence=fact,
                label=ConflictLabel.NO_CONFLICT,
                relations=["HasA", "HasProperty"],
                difficulty="easy"
            )
        
        # =====================================================================
        # TASK 4: Exceptions (CORE TASK) - 40 samples
        # =====================================================================
        t4 = dataset.add_task(
            name="T4: Exceptions",
            description="Exceptions to general rules - the CORE testing task"
        )
        
        # Flightless birds (10)
        exception_birds = [
            ("Penguins cannot fly.", ConflictLabel.NO_CONFLICT, []),
            ("Ostriches cannot fly.", ConflictLabel.NO_CONFLICT, []),
            ("Emus cannot fly.", ConflictLabel.NO_CONFLICT, []),
            ("Kiwis cannot fly.", ConflictLabel.NO_CONFLICT, []),
            ("Cassowaries cannot fly.", ConflictLabel.NO_CONFLICT, []),
            ("Penguins can swim underwater.", ConflictLabel.NO_CONFLICT, []),
            ("Ostriches can run very fast.", ConflictLabel.NO_CONFLICT, []),
            ("Kiwis are nocturnal birds.", ConflictLabel.NO_CONFLICT, []),
            ("Penguins live in cold climates.", ConflictLabel.NO_CONFLICT, []),
            ("Flightless birds have wings but cannot fly.", ConflictLabel.NO_CONFLICT, []),
        ]
        
        # Electric car exceptions (10)
        exception_vehicles = [
            ("Electric cars do not use gasoline.", ConflictLabel.NO_CONFLICT, []),
            ("Electric cars do not produce exhaust emissions.", ConflictLabel.NO_CONFLICT, []),
            ("Electric cars do not have traditional engines.", ConflictLabel.NO_CONFLICT, []),
            ("Bicycles do not use fuel.", ConflictLabel.NO_CONFLICT, []),
            ("Sailboats do not use engines.", ConflictLabel.NO_CONFLICT, []),
            ("Electric cars use batteries for power.", ConflictLabel.NO_CONFLICT, []),
            ("Hybrid cars use both fuel and electricity.", ConflictLabel.NO_CONFLICT, []),
            ("Gliders do not have engines.", ConflictLabel.NO_CONFLICT, []),
            ("Rowboats do not use motors.", ConflictLabel.NO_CONFLICT, []),
            ("Solar cars use solar energy.", ConflictLabel.NO_CONFLICT, []),
        ]
        
        # Fish exceptions (10)
        exception_fish = [
            ("Lungfish can breathe air.", ConflictLabel.NO_CONFLICT, []),
            ("Flying fish can glide above water.", ConflictLabel.NO_CONFLICT, []),
            ("Mudskippers can move on land.", ConflictLabel.NO_CONFLICT, []),
            ("Eels can travel short distances on land.", ConflictLabel.NO_CONFLICT, []),
            ("Some fish can survive out of water briefly.", ConflictLabel.NO_CONFLICT, []),
            ("Climbing perch can move between ponds.", ConflictLabel.NO_CONFLICT, []),
            ("Sharks do not have swim bladders.", ConflictLabel.NO_CONFLICT, []),
            ("Some fish give birth to live young.", ConflictLabel.NO_CONFLICT, []),
            ("Seahorses have males carry eggs.", ConflictLabel.NO_CONFLICT, []),
            ("Electric eels can generate electricity.", ConflictLabel.NO_CONFLICT, []),
        ]
        
        # Mammal exceptions (10)
        exception_mammals = [
            ("Platypuses lay eggs despite being mammals.", ConflictLabel.NO_CONFLICT, []),
            ("Echidnas lay eggs despite being mammals.", ConflictLabel.NO_CONFLICT, []),
            ("Bats are the only mammals that can truly fly.", ConflictLabel.NO_CONFLICT, []),
            ("Whales live entirely in water.", ConflictLabel.NO_CONFLICT, []),
            ("Dolphins breathe air but live in water.", ConflictLabel.NO_CONFLICT, []),
            ("Naked mole rats are cold-blooded mammals.", ConflictLabel.NO_CONFLICT, []),
            ("Some mammals like dolphins sleep with half their brain.", ConflictLabel.NO_CONFLICT, []),
            ("Humans walk upright on two legs.", ConflictLabel.NO_CONFLICT, []),
            ("Seals can hold their breath underwater for long periods.", ConflictLabel.NO_CONFLICT, []),
            ("Pangolins have scales instead of fur.", ConflictLabel.NO_CONFLICT, []),
        ]
        
        for sentence, label, conflicts in exception_birds + exception_vehicles + exception_fish + exception_mammals:
            t4.add_sample(
                sentence=sentence,
                label=label,
                conflicts_with=conflicts,
                conflict_type=ConflictCategory.EXCEPTION_VIOLATION if label == ConflictLabel.CONFLICT else ConflictCategory.NONE,
                difficulty="medium"
            )
        
        # =====================================================================
        # TASK 5: Direct Contradictions (INTENTIONAL) - 40 samples
        # =====================================================================
        t5 = dataset.add_task(
            name="T5: Direct Contradictions",
            description="Intentional conflicts with established knowledge"
        )
        
        # These directly contradict T1-T4 (20 conflicts + 20 non-conflicts)
        contradictions = [
            # Conflicting statements (20)
            ("Penguins can fly.", ConflictLabel.CONFLICT, ["Penguins cannot fly."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Ostriches can fly.", ConflictLabel.CONFLICT, ["Ostriches cannot fly."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Electric cars use petrol.", ConflictLabel.CONFLICT, ["Electric cars do not use gasoline."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Fish breathe air through lungs.", ConflictLabel.CONFLICT, ["Fish breathe through gills."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Birds have scales.", ConflictLabel.CONFLICT, ["Birds have feathers."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Cats can bark.", ConflictLabel.CONFLICT, ["Dogs are mammals that can bark.", "Cats are mammals that can meow."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Snakes have legs.", ConflictLabel.CONFLICT, ["Snakes are legless reptiles."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Whales are fish.", ConflictLabel.CONFLICT, ["Whales are aquatic mammals."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Bicycles use gasoline.", ConflictLabel.CONFLICT, ["Bicycles are powered by humans."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Fire is cold.", ConflictLabel.CONFLICT, ["Fire produces heat."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Humans can breathe underwater without equipment.", ConflictLabel.CONFLICT, ["Humans breathe air through lungs."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Sharks are mammals.", ConflictLabel.CONFLICT, ["Sharks are fish."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Airplanes travel underwater.", ConflictLabel.CONFLICT, ["Airplanes can fly."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Ice is hot.", ConflictLabel.CONFLICT, ["Ice is frozen water and is cold."], ConflictCategory.DIRECT_CONTRADICTION),
            ("The sun produces darkness.", ConflictLabel.CONFLICT, ["The sun produces light."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Trees breathe through gills.", ConflictLabel.CONFLICT, ["Trees photosynthesize."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Rocks are living organisms.", ConflictLabel.CONFLICT, ["Rocks are non-living."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Water is dry.", ConflictLabel.CONFLICT, ["Water is wet."], ConflictCategory.DIRECT_CONTRADICTION),
            ("Spiders have six legs.", ConflictLabel.CONFLICT, ["Spiders have eight legs."], ConflictCategory.DIRECT_CONTRADICTION),
            ("The moon produces its own light.", ConflictLabel.CONFLICT, ["The moon reflects sunlight."], ConflictCategory.DIRECT_CONTRADICTION),
            
            # Non-conflicting statements (20)
            ("Eagles have sharp talons.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Salmon swim upstream to spawn.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Trains are more efficient than cars for long distances.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Dolphins are intelligent creatures.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Cacti store water in their stems.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Owls are nocturnal hunters.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Pandas eat bamboo.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Deserts receive very little rainfall.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Mountains are formed by tectonic activity.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Bees collect nectar from flowers.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Crocodiles are cold-blooded reptiles.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Hurricanes form over warm ocean waters.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Diamonds are the hardest natural material.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Lightning is an electrical discharge.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Giraffes have long necks.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Volcanoes erupt molten lava.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Kangaroos carry their young in pouches.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Camels store fat in their humps.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Polar bears have white fur.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
            ("Zebras have black and white stripes.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE),
        ]
        
        for sentence, label, conflicts, conflict_type in contradictions:
            t5.add_sample(
                sentence=sentence,
                label=label,
                conflicts_with=conflicts,
                conflict_type=conflict_type,
                difficulty="medium" if label == ConflictLabel.NO_CONFLICT else "hard"
            )
        
        # =====================================================================
        # TASK 6: Paraphrase & QA Conflicts - 40 samples
        # =====================================================================
        t6 = dataset.add_task(
            name="T6: Paraphrase & QA Conflicts",
            description="Same knowledge in different surface forms"
        )
        
        paraphrase_samples = [
            # Conflict paraphrases (20)
            ("Can penguins fly?", ConflictLabel.CONFLICT, ["Penguins cannot fly."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Are penguins capable of flight?", ConflictLabel.CONFLICT, ["Penguins cannot fly."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Do penguins have the ability to fly?", ConflictLabel.CONFLICT, ["Penguins cannot fly."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Penguins possess the capability to fly.", ConflictLabel.CONFLICT, ["Penguins cannot fly."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Flying is something penguins can do.", ConflictLabel.CONFLICT, ["Penguins cannot fly."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Do electric cars run on gasoline?", ConflictLabel.CONFLICT, ["Electric cars do not use gasoline."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Electric vehicles use petrol as fuel.", ConflictLabel.CONFLICT, ["Electric cars do not use gasoline."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Can fish walk on land?", ConflictLabel.CONFLICT, ["Fish cannot walk on land."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Fish have the ability to walk.", ConflictLabel.CONFLICT, ["Most fish cannot survive on land."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Is it possible for cats to bark?", ConflictLabel.CONFLICT, ["Cats are mammals that can meow."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Cats possess barking capabilities.", ConflictLabel.CONFLICT, ["Dogs are mammals that can bark."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Can humans breathe water?", ConflictLabel.CONFLICT, ["Humans breathe air through lungs."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Breathing underwater is possible for humans without equipment.", ConflictLabel.CONFLICT, ["Humans cannot breathe underwater without equipment."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Do snakes possess legs?", ConflictLabel.CONFLICT, ["Snakes are legless reptiles."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Snakes have limbs for walking.", ConflictLabel.CONFLICT, ["Snakes do not have legs."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Are whales a type of fish?", ConflictLabel.CONFLICT, ["Whales are aquatic mammals."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Whales belong to the fish family.", ConflictLabel.CONFLICT, ["Whales are cetaceans."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Can ostriches take flight?", ConflictLabel.CONFLICT, ["Ostriches cannot fly."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Ostriches are capable of flying.", ConflictLabel.CONFLICT, ["Ostriches cannot fly."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            ("Do sharks breathe air?", ConflictLabel.CONFLICT, ["Sharks have gills."], ConflictCategory.PARAPHRASE_CONFLICT, "hard"),
            
            # Non-conflict paraphrases (20)
            ("Can birds fly?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Are birds capable of flight?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Do fish live in water?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Fish inhabit aquatic environments.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Can dogs bark?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Dogs have the ability to bark.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Are mammals warm-blooded?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Mammals maintain constant body temperature.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Do vehicles need energy?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Vehicles require power to operate.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Can cats meow?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Cats possess the ability to meow.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Do eagles have wings?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Eagles possess wings for flight.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Are sharks fish?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Sharks belong to the fish category.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Do bicycles have two wheels?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Bicycles are equipped with two wheels.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Can elephants breathe air?", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
            ("Elephants breathe through their lungs.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "easy"),
        ]
        
        for sentence, label, conflicts, conflict_type, difficulty in paraphrase_samples:
            t6.add_sample(
                sentence=sentence,
                label=label,
                conflicts_with=conflicts,
                conflict_type=conflict_type,
                difficulty=difficulty
            )
        
        # =====================================================================
        # TASK 7: Multi-hop Logical Reasoning - 40 samples
        # =====================================================================
        t7 = dataset.add_task(
            name="T7: Multi-hop Logical Reasoning",
            description="Requires combining multiple facts for conflict detection"
        )
        
        # Multi-hop conflicts (20)
        multihop_conflicts = [
            ("Penguins can fly because they are birds.", ConflictLabel.CONFLICT, 
             ["Birds can fly.", "Penguins are birds.", "But penguins cannot fly."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Birds can fly (T1)", "2. Penguins are birds (T2)", "3. Therefore penguins should fly", "4. BUT penguins cannot fly (T4) - CONFLICT!"],
             "hard"),
            
            ("Since ostriches are birds, they must be able to fly.", ConflictLabel.CONFLICT,
             ["Birds can fly.", "Ostriches are birds.", "But ostriches cannot fly."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Birds can fly (T1)", "2. Ostriches are birds (T2)", "3. Therefore ostriches should fly", "4. BUT ostriches cannot fly (T4) - CONFLICT!"],
             "hard"),
            
            ("Electric cars use fuel because all vehicles need energy.", ConflictLabel.CONFLICT,
             ["Vehicles need energy.", "Electric cars are vehicles.", "But electric cars don't use fuel."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Vehicles need energy (T1)", "2. Electric cars are vehicles (T2)", "3. BUT electric cars use electricity, not fuel (T4) - CONFLICT!"],
             "hard"),
            
            ("Sharks can walk on land since some fish can move on land.", ConflictLabel.CONFLICT,
             ["Some fish like mudskippers can move on land.", "Sharks are fish.", "But sharks cannot walk on land."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Some fish can move on land (exception)", "2. Sharks are fish (T2)", "3. BUT sharks cannot walk on land - overgeneralization CONFLICT!"],
             "hard"),
            
            ("Whales breathe through gills because they live in water.", ConflictLabel.CONFLICT,
             ["Fish breathe through gills.", "Whales live in water.", "But whales breathe air."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Fish breathe through gills (T1)", "2. Whales live in water", "3. BUT whales are mammals and breathe air (T3) - CONFLICT!"],
             "hard"),
            
            ("Platypuses give birth to live young since they are mammals.", ConflictLabel.CONFLICT,
             ["Mammals give birth to live young.", "Platypuses are mammals.", "But platypuses lay eggs."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Mammals give birth to live young (T1)", "2. Platypuses are mammals (T2)", "3. BUT platypuses lay eggs (T4) - exception CONFLICT!"],
             "hard"),
            
            ("Bats cannot fly because mammals cannot fly.", ConflictLabel.CONFLICT,
             ["Most mammals cannot fly.", "Bats are mammals.", "But bats can fly."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Most mammals cannot fly", "2. Bats are mammals (T2)", "3. BUT bats can fly (T4) - exception CONFLICT!"],
             "hard"),
            
            ("Dolphins breathe through gills since they live underwater.", ConflictLabel.CONFLICT,
             ["Fish breathe through gills.", "Dolphins live underwater.", "But dolphins breathe air."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Aquatic animals often have gills", "2. Dolphins live underwater", "3. BUT dolphins are mammals and breathe air - CONFLICT!"],
             "hard"),
            
            ("All birds build nests, so penguins must build nests in trees.", ConflictLabel.CONFLICT,
             ["Most birds build nests.", "Penguins are birds.", "But penguins don't build nests in trees."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Most birds build nests (T1)", "2. Penguins are birds (T2)", "3. BUT penguins live in Antarctica with no trees - CONFLICT!"],
             "hard"),
            
            ("Hybrid cars don't need energy since they don't use pure fuel.", ConflictLabel.CONFLICT,
             ["Vehicles need energy.", "Hybrid cars are vehicles.", "Hybrid cars use both fuel and electricity."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. All vehicles need energy (T1)", "2. Hybrid cars are vehicles (T2)", "3. Hybrid cars DO need energy (hybrid means two sources) - CONFLICT!"],
             "hard"),
            
            ("Eels can fly because some fish can leave water.", ConflictLabel.CONFLICT,
             ["Flying fish can glide above water.", "Eels are fish.", "But eels cannot fly."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Some fish (flying fish) can glide", "2. Eels are fish", "3. BUT eels cannot fly - overgeneralization CONFLICT!"],
             "hard"),
            
            ("Since birds have feathers, penguins must be able to use them for flight.", ConflictLabel.CONFLICT,
             ["Birds have feathers.", "Penguins have feathers.", "But penguins cannot fly."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Birds have feathers (T1)", "2. Penguins have feathers (T3)", "3. Feathers enable flight", "4. BUT penguins cannot fly (T4) - CONFLICT!"],
             "hard"),
            
            ("Lungfish breathe only through gills like all fish.", ConflictLabel.CONFLICT,
             ["Fish breathe through gills.", "Lungfish are fish.", "But lungfish can breathe air."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Fish breathe through gills (T1)", "2. Lungfish are fish (T2)", "3. BUT lungfish can breathe air (T4) - exception CONFLICT!"],
             "hard"),
            
            ("Echidnas give birth to live young because they are mammals.", ConflictLabel.CONFLICT,
             ["Mammals give birth to live young.", "Echidnas are mammals.", "But echidnas lay eggs."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Mammals give birth to live young (T1)", "2. Echidnas are mammals (T2)", "3. BUT echidnas lay eggs (T4) - exception CONFLICT!"],
             "hard"),
            
            ("Electric vehicles produce exhaust because they are vehicles.", ConflictLabel.CONFLICT,
             ["Most vehicles produce exhaust.", "Electric cars are vehicles.", "But electric cars don't produce exhaust."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Traditional vehicles produce exhaust", "2. Electric cars are vehicles (T2)", "3. BUT electric cars don't produce exhaust (T4) - CONFLICT!"],
             "hard"),
            
            ("Seahorses are fast swimmers since they are fish.", ConflictLabel.CONFLICT,
             ["Fish can swim.", "Seahorses are fish.", "But seahorses are slow swimmers."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Fish can swim (T1)", "2. Seahorses are fish (T2)", "3. Implied: seahorses swim fast", "4. BUT seahorses are actually slow - CONFLICT!"],
             "hard"),
            
            ("Kiwis can fly short distances because they have wings.", ConflictLabel.CONFLICT,
             ["Birds with wings can fly.", "Kiwis have wings.", "But kiwis cannot fly."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Birds with wings can fly (T1)", "2. Kiwis have wings (T3)", "3. Therefore kiwis should fly", "4. BUT kiwis cannot fly (T4) - CONFLICT!"],
             "hard"),
            
            ("Snakes can breathe underwater since they are cold-blooded.", ConflictLabel.CONFLICT,
             ["Fish are cold-blooded and live in water.", "Snakes are cold-blooded.", "But snakes breathe air."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Cold-blooded aquatic animals often have gills", "2. Snakes are cold-blooded reptiles", "3. BUT snakes breathe air, not underwater - CONFLICT!"],
             "hard"),
            
            ("Sailboats use fuel since they are vehicles.", ConflictLabel.CONFLICT,
             ["Vehicles need energy.", "Sailboats are vehicles.", "But sailboats use wind, not fuel."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Vehicles need energy (T1)", "2. Sailboats are vehicles (T2)", "3. Implied: use fuel", "4. BUT sailboats use wind energy (T4) - CONFLICT!"],
             "hard"),
            
            ("All mammals have fur, so whales must have fur.", ConflictLabel.CONFLICT,
             ["Mammals have fur or hair.", "Whales are mammals.", "But whales don't have fur."],
             ConflictCategory.MULTIHOP_REASONING,
             ["1. Mammals have fur/hair (T1)", "2. Whales are mammals (T2)", "3. Therefore whales should have fur", "4. BUT whales lost fur evolutionarily - CONFLICT!"],
             "hard"),
        ]
        
        # Non-conflict multi-hop (20)
        multihop_valid = [
            ("Sparrows can fly because they are birds.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Birds can fly (T1)", "2. Sparrows are birds (T2)", "3. Therefore sparrows can fly - VALID!"], "medium"),
            
            ("Eagles have feathers since they are birds.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Birds have feathers (T1)", "2. Eagles are birds (T2)", "3. Therefore eagles have feathers - VALID!"], "medium"),
            
            ("Sharks can swim because they are fish.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Fish can swim (T1)", "2. Sharks are fish (T2)", "3. Therefore sharks can swim - VALID!"], "medium"),
            
            ("Salmon live in water since they are fish.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Fish live in water (T1)", "2. Salmon are fish (T2)", "3. Therefore salmon live in water - VALID!"], "medium"),
            
            ("Dogs are warm-blooded because they are mammals.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Mammals are warm-blooded (T1)", "2. Dogs are mammals (T2)", "3. Therefore dogs are warm-blooded - VALID!"], "medium"),
            
            ("Cats have fur since they are mammals.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Mammals have fur/hair (T1)", "2. Cats are mammals (T2)", "3. Therefore cats have fur - VALID!"], "medium"),
            
            ("SUVs need energy because they are vehicles.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Vehicles need energy (T1)", "2. SUVs are vehicles (T2)", "3. Therefore SUVs need energy - VALID!"], "medium"),
            
            ("Sports cars have wheels since they are vehicles.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Vehicles have wheels (T1)", "2. Sports cars are vehicles (T2)", "3. Therefore sports cars have wheels - VALID!"], "medium"),
            
            ("Tuna have gills because they are fish.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Fish have gills (T1)", "2. Tuna are fish (T2)", "3. Therefore tuna have gills - VALID!"], "medium"),
            
            ("Owls have beaks since they are birds.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Birds have beaks (T1)", "2. Owls are birds (T2)", "3. Therefore owls have beaks - VALID!"], "medium"),
            
            ("Lions are warm-blooded because they are mammals.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Mammals are warm-blooded (T1)", "2. Lions are mammals", "3. Therefore lions are warm-blooded - VALID!"], "medium"),
            
            ("Parrots lay eggs since they are birds.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Birds lay eggs (T1)", "2. Parrots are birds (T2)", "3. Therefore parrots lay eggs - VALID!"], "medium"),
            
            ("Goldfish have fins because they are fish.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Fish have fins (T1)", "2. Goldfish are fish (T2)", "3. Therefore goldfish have fins - VALID!"], "medium"),
            
            ("Elephants breathe air since they are mammals.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Mammals breathe air (T1)", "2. Elephants are mammals (T2)", "3. Therefore elephants breathe air - VALID!"], "medium"),
            
            ("Ducks have wings because they are birds.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Birds have wings (T1)", "2. Ducks are birds (T2)", "3. Therefore ducks have wings - VALID!"], "medium"),
            
            ("Motorcycles need energy since they are vehicles.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Vehicles need energy (T1)", "2. Motorcycles are vehicles (T2)", "3. Therefore motorcycles need energy - VALID!"], "medium"),
            
            ("Barracudas are cold-blooded because they are fish.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Fish are cold-blooded (T1)", "2. Barracudas are fish (T2)", "3. Therefore barracudas are cold-blooded - VALID!"], "medium"),
            
            ("Ravens have feathers since they are birds.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Birds have feathers (T1)", "2. Ravens are birds (T2)", "3. Therefore ravens have feathers - VALID!"], "medium"),
            
            ("Flamingos can fly because they are birds.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Birds can fly (T1)", "2. Flamingos are birds (T2)", "3. Therefore flamingos can fly - VALID!"], "medium"),
            
            ("Hummingbirds have beaks since they are birds.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE,
             ["1. Birds have beaks (T1)", "2. Hummingbirds are birds (T2)", "3. Therefore hummingbirds have beaks - VALID!"], "medium"),
        ]
        
        for sentence, label, conflicts, conflict_type, reasoning, difficulty in multihop_conflicts:
            t7.add_sample(
                sentence=sentence,
                label=label,
                conflicts_with=conflicts,
                conflict_type=conflict_type,
                reasoning_chain=reasoning,
                difficulty=difficulty
            )
        
        for sentence, label, conflicts, conflict_type, reasoning, difficulty in multihop_valid:
            t7.add_sample(
                sentence=sentence,
                label=label,
                conflicts_with=conflicts,
                conflict_type=conflict_type,
                reasoning_chain=reasoning,
                difficulty=difficulty
            )
        
        # =====================================================================
        # TASK 8: Delayed Contradictions (HARDEST) - 40 samples
        # =====================================================================
        t8 = dataset.add_task(
            name="T8: Delayed Contradictions",
            description="Conflicts that appear after several tasks - tests long-term memory"
        )
        
        # These reference facts from T1-T2 but conflict appears much later (20 conflicts + 20 non-conflicts)
        delayed_conflicts = [
            # Delayed conflicts (20) - ambiguous/hard
            ("Penguins can soar through the sky.", ConflictLabel.AMBIGUOUS, 
             ["Penguins are birds (T2)", "Birds can fly (T1)", "Penguins cannot fly (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Electric cars refuel at gas stations.", ConflictLabel.AMBIGUOUS,
             ["Electric cars are vehicles (T2)", "Vehicles need energy (T1)", "Electric cars don't use gasoline (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Fish walk on beaches.", ConflictLabel.AMBIGUOUS,
             ["Fish live in water (T1)", "Some fish can move on land (T4)", "But most fish cannot walk"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Ostriches take flight to escape predators.", ConflictLabel.AMBIGUOUS,
             ["Ostriches are birds (T2)", "Birds can fly (T1)", "Ostriches cannot fly (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Whales use gills to breathe underwater.", ConflictLabel.AMBIGUOUS,
             ["Whales live in water", "Fish breathe through gills (T1)", "Whales breathe air (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Bicycles fill up on diesel fuel.", ConflictLabel.AMBIGUOUS,
             ["Bicycles are vehicles (T2)", "Vehicles need energy (T1)", "Bicycles are human-powered (T1)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Cats bark at strangers.", ConflictLabel.AMBIGUOUS,
             ["Cats are animals", "Dogs bark (T1)", "Cats meow (T1)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Platypuses nurse their young with milk after hatching eggs.", ConflictLabel.AMBIGUOUS,
             ["Platypuses are mammals (T2)", "Mammals give birth to live young (T1)", "Platypuses lay eggs (T4)", "But they do produce milk"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Hybrid cars never need fuel.", ConflictLabel.AMBIGUOUS,
             ["Hybrid cars are vehicles (T2)", "Hybrid cars use both fuel and electricity (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Emus fly south for winter.", ConflictLabel.AMBIGUOUS,
             ["Emus are birds (T2)", "Birds can fly (T1)", "Emus cannot fly (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Sharks walk on the ocean floor.", ConflictLabel.AMBIGUOUS,
             ["Sharks are fish (T2)", "Fish swim (T1)", "Fish cannot walk (T1)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Kiwis migrate by flying.", ConflictLabel.AMBIGUOUS,
             ["Kiwis are birds (T2)", "Birds can fly (T1)", "Kiwis cannot fly (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Echidnas give birth in nests.", ConflictLabel.AMBIGUOUS,
             ["Echidnas are mammals (T2)", "Mammals give birth to live young (T1)", "Echidnas lay eggs (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Bats cannot fly because they are mammals.", ConflictLabel.AMBIGUOUS,
             ["Bats are mammals (T2)", "Mammals don't fly (general)", "Bats can fly (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Lungfish only survive underwater.", ConflictLabel.AMBIGUOUS,
             ["Lungfish are fish (T2)", "Fish live in water (T1)", "Lungfish can breathe air (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Electric vehicles emit carbon dioxide.", ConflictLabel.AMBIGUOUS,
             ["Electric cars are vehicles (T2)", "Vehicles produce emissions (traditional)", "Electric cars don't produce exhaust (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Cassowaries soar at high altitudes.", ConflictLabel.AMBIGUOUS,
             ["Cassowaries are birds", "Birds can fly (T1)", "Cassowaries cannot fly (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Mudskippers breathe only through gills.", ConflictLabel.AMBIGUOUS,
             ["Mudskippers are fish (T2)", "Fish breathe through gills (T1)", "Mudskippers can move on land (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Sailboats consume diesel.", ConflictLabel.AMBIGUOUS,
             ["Sailboats are vehicles (T2)", "Vehicles need energy (T1)", "Sailboats use wind (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            ("Naked mole rats seek warm environments.", ConflictLabel.AMBIGUOUS,
             ["Naked mole rats are mammals (T2)", "Mammals are warm-blooded (T1)", "Naked mole rats are cold-blooded (T4)"],
             ConflictCategory.DELAYED_CONFLICT, "hard"),
            
            # Valid delayed references (20)
            ("Eagles still have sharp vision many tasks later.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Salmon continue to swim upstream even in task 8.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Dogs remain capable of barking across all tasks.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Dolphins are still intelligent in later tasks.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Owls hunt at night regardless of task order.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Pandas eat bamboo consistently through all tasks.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Tuna swim in the ocean in every task.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Cars have four wheels from task 1 to task 8.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Giraffes have long necks in all tasks.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Volcanoes erupt lava throughout the dataset.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Bees collect nectar consistently.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Crocodiles remain cold-blooded reptiles.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Lightning remains electrical across tasks.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Diamonds stay the hardest material.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Kangaroos still carry young in pouches.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Camels continue to store fat in humps.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Polar bears have white fur throughout.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Zebras maintain their stripes in all tasks.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Hurricanes form over warm waters consistently.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
            ("Cacti store water in stems across the dataset.", ConflictLabel.NO_CONFLICT, [], ConflictCategory.NONE, "medium"),
        ]
        
        for sentence, label, conflicts, conflict_type, difficulty in delayed_conflicts:
            t8.add_sample(
                sentence=sentence,
                label=label,
                conflicts_with=conflicts,
                conflict_type=conflict_type,
                difficulty=difficulty
            )
        
        return dataset


# =============================================================================
# Utility Functions
# =============================================================================

def print_publication_summary(dataset: SeCAPublicationDataset) -> None:
    """Print publication-ready summary."""
    stats = dataset.get_statistics()
    
    print("=" * 70)
    print(f"  {stats['name']} (v{stats['version']})")
    print("=" * 70)
    print()
    print(f"  Total Tasks: {stats['total_tasks']}")
    print(f"  Samples per Task: {stats['samples_per_task']}")
    print(f"  Total Samples: {stats['total_samples']}")
    print()
    print("  LABEL DISTRIBUTION:")
    for label, count in stats['label_distribution'].items():
        pct = (count / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0
        print(f"    - {label}: {count} ({pct:.1f}%)")
    print()
    print("  CONFLICT TYPES:")
    for ctype, count in stats['conflict_types'].items():
        print(f"    - {ctype}: {count}")
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\nCreating SeCA Publication Dataset (320 samples)...")
    print()
    
    # Create full dataset
    dataset = SeCAPublicationDataset.create_publication_dataset()
    
    # Print summary
    print_publication_summary(dataset)
    
    # Save to file
    output_path = Path(__file__).parent / "seca_publication_dataset.json"
    dataset.save(str(output_path))
    print(f"  Dataset saved to: {output_path}")
    print()
    print("=" * 70)
    print("  PUBLICATION DATASET READY")
    print("=" * 70)

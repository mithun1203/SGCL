"""
SeCA Dataset Loader for SG-CL Training

Loads SeCA v2.0 dataset in task-sequential format for continual learning experiments.
Each task represents a different drift scenario.

Features:
- Dynamic 80/20 train/test split
- Reproducible with seed
- Task-sequential loading
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SeCATask:
    """A single SeCA task."""
    task_id: int
    task_name: str
    drift_type: str
    samples: List[str]
    description: str


class SeCALoader:
    """
    Loads SeCA dataset in task-sequential format.
    
    SeCA has 8 tasks:
    1. exception_overwriting
    2. category_confusion
    3. attribute_transfer
    4. hierarchy_violation
    5. over_generalization
    6. context_shift
    7. temporal_drift
    8. delayed_drift
    """
    
    def __init__(self, seca_path: str = "sid/seca_10k_dataset.json"):
        """
        Initialize SeCA loader.
        
        Args:
            seca_path: Path to SeCA JSON file (default: 10k augmented dataset)
        """
        self.seca_path = Path(seca_path)
        self.tasks: List[SeCATask] = []
        self._load_seca()
    
    def _load_seca(self):
        """Load SeCA dataset from JSON."""
        if not self.seca_path.exists():
            raise FileNotFoundError(f"SeCA dataset not found: {self.seca_path}")
        
        with open(self.seca_path, 'r') as f:
            data = json.load(f)
        
        dataset_info = data.get('dataset_info', {})
        tasks_data = data.get('tasks', {})
        
        print(f"Loading SeCA v{dataset_info.get('version', 'Unknown')}")
        print(f"Total samples: {dataset_info.get('total_samples', 0)}")
        print(f"Tasks: {len(tasks_data)}")
        
        # Parse tasks
        for task_id, (task_name, task_info) in enumerate(tasks_data.items()):
            samples = []
            
            # Extract samples from scenarios
            for scenario in task_info.get('scenarios', []):
                samples.append(scenario['input'])
                samples.append(scenario['conflict'])
                
                # Add supporting facts if available
                if 'supporting_facts' in scenario:
                    samples.extend(scenario['supporting_facts'])
            
            self.tasks.append(SeCATask(
                task_id=task_id,
                task_name=task_name,
                drift_type=task_info.get('type', 'unknown'),
                samples=samples,
                description=task_info.get('description', '')
            ))
    
    def get_tasks(self) -> List[List[str]]:
        """
        Get tasks as list of sample lists (for training).
        
        Returns:
            List[List[str]] where each inner list is a task
        """
        return [task.samples for task in self.tasks]
    
    def get_task_names(self) -> List[str]:
        """Get task names."""
        return [task.task_name for task in self.tasks]
    
    def get_task(self, task_id: int) -> SeCATask:
        """Get specific task by ID."""
        if 0 <= task_id < len(self.tasks):
            return self.tasks[task_id]
        raise IndexError(f"Task {task_id} not found")
    
    def get_tasks_by_drift_type(self, drift_type: str) -> List[SeCATask]:
        """Get tasks matching specific drift type."""
        return [task for task in self.tasks if task.drift_type == drift_type]
    
    def get_subset(self, task_ids: List[int]) -> Tuple[List[List[str]], List[str]]:
        """
        Get subset of tasks.
        
        Args:
            task_ids: List of task IDs to include
        
        Returns:
            (tasks, task_names) tuple
        """
        tasks = [self.tasks[i].samples for i in task_ids if 0 <= i < len(self.tasks)]
        names = [self.tasks[i].task_name for i in task_ids if 0 <= i < len(self.tasks)]
        return tasks, names
    
    def print_summary(self):
        """Print dataset summary."""
        print("\n" + "="*70)
        print("  SeCA Dataset Summary")
        print("="*70)
        
        total_samples = sum(len(task.samples) for task in self.tasks)
        print(f"\nTotal Tasks: {len(self.tasks)}")
        print(f"Total Samples: {total_samples}")
        
        print("\nTasks:")
        for task in self.tasks:
            print(f"  {task.task_id + 1}. {task.task_name}")
            print(f"     Type: {task.drift_type}")
            print(f"     Samples: {len(task.samples)}")
            print(f"     Description: {task.description[:60]}...")
        
        print("="*70 + "\n")


def create_toy_tasks() -> Tuple[List[List[str]], List[str]]:
    """
    Create toy sequential tasks for quick testing.
    
    Returns:
        (tasks, task_names) tuple
    """
    tasks = [
        # Task 1: General bird facts
        [
            "Birds can fly.",
            "Birds have wings.",
            "Birds have feathers.",
            "Eagles are powerful birds.",
            "Sparrows are small birds."
        ],
        
        # Task 2: Penguin exception (conflicts with Task 1)
        [
            "Penguins cannot fly.",  # Conflicts with "Birds can fly"
            "Penguins are birds.",
            "Penguins live in Antarctica.",
            "Penguins are excellent swimmers.",
            "Penguins have flippers instead of wings."
        ],
        
        # Task 3: More bird types
        [
            "Ostriches are large birds.",
            "Ostriches cannot fly.",  # Another exception
            "Robins can fly.",
            "Hawks are birds of prey.",
            "Hummingbirds are very small."
        ]
    ]
    
    task_names = [
        "General Bird Facts",
        "Penguin Exception",
        "Mixed Bird Types"
    ]
    
    return tasks, task_names


def create_minimal_tasks() -> Tuple[List[List[str]], List[str]]:
    """
    Create minimal tasks for fastest testing (2-3 samples per task).
    
    Returns:
        (tasks, task_names) tuple
    """
    tasks = [
        ["Birds can fly.", "Eagles have sharp talons."],
        ["Penguins cannot fly.", "Penguins are birds."]
    ]
    
    task_names = ["Task 1", "Task 2"]
    
    return tasks, task_names


# ══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ══════════════════════════════════════════════════════════════════════════════

def load_seca_tasks(
    seca_path: str = "sid/seca_10k_dataset.json"
) -> Tuple[List[List[str]], List[str]]:
    """
    Load full SeCA 10k dataset for training.
    
    Returns:
        (tasks, task_names) tuple ready for SGCLTrainer
    
    Example:
        >>> tasks, names = load_seca_tasks()
        >>> trainer = SGCLTrainer(config)
        >>> trainer.train_on_tasks(tasks, names)
    """
    try:
        loader = SeCALoader(seca_path)
        return loader.get_tasks(), loader.get_task_names()
    except FileNotFoundError:
        print(f"WARNING: SeCA file not found at {seca_path}")
        print("Falling back to toy tasks for demonstration")
        return create_toy_tasks()


def load_seca_for_training(
    seca_path: str = "sid/seca_10k_dataset.json",
    subset: Optional[List[int]] = None,
    train_split: float = 0.8,
    seed: int = 42
) -> Tuple[List[List[str]], List[List[str]], List[str]]:
    """
    Load SeCA 10k dataset with 80/20 train/test split.
    
    Args:
        seca_path: Path to SeCA JSON
        subset: Optional list of task IDs to use
        train_split: Train split ratio (default 0.8 for 80/20)
        seed: Random seed for reproducibility
    
    Returns:
        (train_tasks, test_tasks, task_names) tuple ready for training
    
    Example:
        >>> train, test, names = load_seca_for_training()
        >>> trainer = SGCLTrainer(config)
        >>> trainer.train_on_tasks(train, names)
        >>> # Then evaluate on test
    """
    loader = SeCALoader(seca_path)
    
    if subset:
        all_tasks, task_names = loader.get_subset(subset)
    else:
        all_tasks, task_names = loader.get_tasks(), loader.get_task_names()
    
    # Split each task into train/test
    random.seed(seed)
    train_tasks = []
    test_tasks = []
    
    for task in all_tasks:
        task_copy = task.copy()
        random.shuffle(task_copy)
        
        split_idx = int(len(task_copy) * train_split)
        train_tasks.append(task_copy[:split_idx])
        test_tasks.append(task_copy[split_idx:])
    
    return train_tasks, test_tasks, task_names


def split_task_data(
    task_samples: List[str],
    train_split: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split a single task's samples into train/test.
    
    Args:
        task_samples: List of samples for one task
        train_split: Train ratio (default 0.8)
        seed: Random seed
    
    Returns:
        (train_samples, test_samples)
    """
    random.seed(seed)
    shuffled = task_samples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_split)
    return shuffled[:split_idx], shuffled[split_idx:]


if __name__ == '__main__':
    # Demo
    print("SeCA Loader Demo\n")
    
    # Try loading real SeCA
    try:
        loader = SeCALoader("sid/seca_publication_v2.json")
        loader.print_summary()
        
        # Show first task
        task = loader.get_task(0)
        print(f"\nFirst Task: {task.task_name}")
        print(f"Samples: {task.samples[:3]}...")
        
    except FileNotFoundError:
        print("SeCA file not found. Using toy tasks instead.\n")
        tasks, names = create_toy_tasks()
        
        print(f"Toy Tasks: {len(tasks)}")
        for i, (task, name) in enumerate(zip(tasks, names)):
            print(f"\n{name} ({len(task)} samples):")
            for sample in task[:3]:
                print(f"  - {sample}")

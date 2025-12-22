"""                                                                                      
SG-CL Pipeline Integration Module
=================================

This module provides the interfaces and classes needed to integrate SID 
into the full Symbolic-Gated Continual Learning (SG-CL) pipeline.

Pipeline Flow:
    New Knowledge → SID (Conflict Detection) → Gating Decision → 
    Guard-Rail Generation → SG-CL Training → Evaluation

Author: Mithun Naik
Project: SGCL Capstone
Version: 1.0.0
License: MIT
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable, Iterator
from datetime import datetime

# Import comprehensive SeCA classes
from .seca_dataset import (
    SeCADataset as RichSeCADataset,
    Sample as SeCASample,
    Task as SeCATask,
    ConflictType as SeCAConflictType,
    create_seca_dataset,
    print_dataset_summary
)

logger = logging.getLogger(__name__)
# ============================================================================
# STEP 1: Sequential Data Interface (SeCA Dataset)
# ============================================================================

@dataclass
class Task:
    """
    Represents a single task in continual learning.
    
    A task contains a set of training samples that arrive together.
    In SG-CL, tasks arrive sequentially, simulating real-world learning.
    """
    task_id: int
    name: str
    samples: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[str]:
        return iter(self.samples)


@dataclass
class TaskBatch:
    """A batch of samples from a task for training."""
    task_id: int
    samples: List[str]
    batch_idx: int
    requires_gating: bool = False
    conflicts: List[Any] = field(default_factory=list)
    guard_rails: List[str] = field(default_factory=list)


class SeCADataset:
    """
    Simple Semantic Consistency Aware Dataset wrapper for pipeline use.
    
    For rich features (conflict types, domains, metadata), use:
        from sid.seca_dataset import SeCADataset as RichSeCADataset
    
    This wrapper provides backward compatibility for simple use cases.
    
    Example:
        >>> dataset = SeCADataset()
        >>> dataset.add_task("Birds can fly", "Sparrows are birds")
        >>> dataset.add_task("Penguins are birds", "Penguins cannot fly")
        >>> 
        >>> for task in dataset:
        ...     print(f"Task {task.task_id}: {len(task)} samples")
    """
    
    def __init__(self):
        self.tasks: List[Task] = []
        self._current_idx = 0
    
    def add_task(self, *samples: str, name: Optional[str] = None, 
                 metadata: Optional[Dict] = None) -> Task:
        """Add a new task with samples."""
        task_id = len(self.tasks)
        task = Task(
            task_id=task_id,
            name=name or f"Task_{task_id}",
            samples=list(samples),
            metadata=metadata or {}
        )
        self.tasks.append(task)
        return task
    
    def add_task_from_list(self, samples: List[str], name: Optional[str] = None) -> Task:
        """Add a task from a list of samples."""
        return self.add_task(*samples, name=name)
    
    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Task:
        return self.tasks[idx]
    
    def get_batches(self, task: Task, batch_size: int = 8) -> Iterator[TaskBatch]:
        """Yield batches from a task."""
        for i in range(0, len(task.samples), batch_size):
            yield TaskBatch(
                task_id=task.task_id,
                samples=task.samples[i:i + batch_size],
                batch_idx=i // batch_size
            )
    
    @classmethod
    def from_rich_dataset(cls, rich_dataset: RichSeCADataset) -> 'SeCADataset':
        """Convert a rich SeCADataset to simple pipeline format."""
        dataset = cls()
        for task in rich_dataset:
            samples = [sample.text for sample in task]
            dataset.add_task(*samples, name=task.name)
        return dataset
    
    @classmethod
    def create_standard(cls) -> 'SeCADataset':
        """Create standard dataset from rich SeCA format."""
        rich_dataset = create_seca_dataset("standard")
        return cls.from_rich_dataset(rich_dataset)


# ============================================================================
# STEP 2 & 3: SID Integration Point
# ============================================================================

class GatingDecision(Enum):
    """Result of the gating decision."""
    NORMAL_TRAINING = "normal"  # No conflict - proceed normally
    GATED_TRAINING = "gated"    # Conflict detected - apply guard-rails
    BLOCK = "block"             # Severe conflict - skip this sample


@dataclass
class SIDResult:
    """
    Result from SID analysis for pipeline integration.
    
    This wraps the ConflictResult with pipeline-specific information.
    """
    sample: str
    has_conflict: bool
    conflict_type: Optional[str] = None
    confidence: float = 0.0
    conflicting_knowledge: List[str] = field(default_factory=list)
    suggested_guard_rails: List[str] = field(default_factory=list)
    gating_decision: GatingDecision = GatingDecision.NORMAL_TRAINING
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample": self.sample,
            "has_conflict": self.has_conflict,
            "conflict_type": self.conflict_type,
            "confidence": self.confidence,
            "gating_decision": self.gating_decision.value,
            "conflicting_knowledge": self.conflicting_knowledge,
            "suggested_guard_rails": self.suggested_guard_rails
        }


class SIDPipelineAdapter:
    """
    Adapter that wraps SID for pipeline integration.
    
    This provides a clean interface between SID and the SG-CL training loop.
    
    Example:
        >>> from sid import create_detector
        >>> from sid.pipeline import SIDPipelineAdapter
        >>> 
        >>> detector = create_detector(offline_only=True)
        >>> sid_adapter = SIDPipelineAdapter(detector)
        >>> 
        >>> result = sid_adapter.analyze("Penguins can fly")
        >>> print(result.gating_decision)  # GatingDecision.GATED_TRAINING
    """
    
    def __init__(self, detector, conflict_threshold: float = 0.5):
        """
        Initialize the SID pipeline adapter.
        
        Args:
            detector: SemanticInconsistencyDetector instance
            conflict_threshold: Minimum confidence to trigger gating
        """
        self.detector = detector
        self.conflict_threshold = conflict_threshold
        self._analysis_history: List[SIDResult] = []
    
    def analyze(self, sample: str) -> SIDResult:
        """
        Analyze a single sample for semantic conflicts.
        
        This is STEP 3 of the pipeline.
        
        Args:
            sample: The text to analyze
            
        Returns:
            SIDResult with conflict info and gating decision
        """
        import time
        start = time.perf_counter()
        
        # Run SID conflict detection
        conflict_result = self.detector.detect_conflict(sample)
        
        processing_time = (time.perf_counter() - start) * 1000
        
        # Extract conflict information
        conflicting_knowledge = []
        suggested_guard_rails = []
        conflict_type = None
        
        if conflict_result.has_conflict:
            for conflict in conflict_result.conflicts:
                # Get the conflicting triple
                if conflict.conflicting_triple:
                    conflicting_knowledge.append(
                        conflict.conflicting_triple.to_natural_language()
                    )
                conflict_type = conflict.conflict_type.value
                
                # Generate basic guard-rail suggestions
                suggested_guard_rails.extend(
                    self._generate_guard_rail_suggestions(conflict)
                )
        
        # Determine gating decision
        if conflict_result.has_conflict:
            confidence = conflict_result.conflicts[0].confidence if conflict_result.conflicts else 0.5
            if confidence >= self.conflict_threshold:
                gating_decision = GatingDecision.GATED_TRAINING
            else:
                gating_decision = GatingDecision.NORMAL_TRAINING
        else:
            gating_decision = GatingDecision.NORMAL_TRAINING
            confidence = 0.0
        
        result = SIDResult(
            sample=sample,
            has_conflict=conflict_result.has_conflict,
            conflict_type=conflict_type,
            confidence=confidence,
            conflicting_knowledge=conflicting_knowledge,
            suggested_guard_rails=suggested_guard_rails,
            gating_decision=gating_decision,
            processing_time_ms=processing_time
        )
        
        self._analysis_history.append(result)
        return result
    
    def analyze_batch(self, samples: List[str]) -> List[SIDResult]:
        """Analyze a batch of samples."""
        return [self.analyze(sample) for sample in samples]
    
    def _generate_guard_rail_suggestions(self, conflict) -> List[str]:
        """
        Generate guard-rail text suggestions based on conflict.
        
        These are suggestions that the GuardRailGenerator can use.
        """
        suggestions = []
        
        if conflict.source_triple and conflict.conflicting_triple:
            source = conflict.source_triple
            conflicting = conflict.conflicting_triple
            
            # Suggest exception handling
            if conflict.conflict_type.value == "direct_contradiction":
                suggestions.append(
                    f"{source.subject} is an exception to the general rule about {source.object}"
                )
                suggestions.append(
                    f"While most {conflicting.object}s can {source.object}, "
                    f"{source.subject} cannot"
                )
        
        return suggestions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        if not self._analysis_history:
            return {"total_analyzed": 0}
        
        conflicts = [r for r in self._analysis_history if r.has_conflict]
        gated = [r for r in self._analysis_history 
                 if r.gating_decision == GatingDecision.GATED_TRAINING]
        
        return {
            "total_analyzed": len(self._analysis_history),
            "conflicts_detected": len(conflicts),
            "conflict_rate": len(conflicts) / len(self._analysis_history),
            "gated_training_rate": len(gated) / len(self._analysis_history),
            "avg_processing_time_ms": sum(r.processing_time_ms for r in self._analysis_history) / len(self._analysis_history)
        }
    
    def reset_history(self):
        """Clear analysis history."""
        self._analysis_history.clear()


# ============================================================================
# STEP 4: Gating Decision Point
# ============================================================================

class GatingController:
    """
    Controls the training path based on SID analysis.
    
    This is STEP 4 of the pipeline - the decision point.
    
    Example:
        >>> controller = GatingController(sid_adapter)
        >>> 
        >>> for batch in task_batches:
        ...     decision = controller.process_batch(batch)
        ...     if decision == GatingDecision.NORMAL_TRAINING:
        ...         normal_train(model, batch)
        ...     else:
        ...         gated_train(model, batch, controller.get_guard_rails())
    """
    
    def __init__(self, sid_adapter: SIDPipelineAdapter,
                 guard_rail_generator: Optional['GuardRailGenerator'] = None):
        self.sid_adapter = sid_adapter
        self.guard_rail_generator = guard_rail_generator
        self._current_guard_rails: List[str] = []
        self._current_conflicts: List[SIDResult] = []
    
    def process_batch(self, batch: TaskBatch) -> GatingDecision:
        """
        Process a batch and determine the training path.
        
        Returns:
            GatingDecision indicating how to train on this batch
        """
        self._current_guard_rails.clear()
        self._current_conflicts.clear()
        
        # Analyze all samples in batch
        results = self.sid_adapter.analyze_batch(batch.samples)
        
        # Check for any conflicts
        conflicts = [r for r in results if r.has_conflict]
        
        if not conflicts:
            batch.requires_gating = False
            return GatingDecision.NORMAL_TRAINING
        
        # Conflicts detected - need gated training
        self._current_conflicts = conflicts
        batch.requires_gating = True
        batch.conflicts = conflicts
        
        # Generate guard-rails if generator available
        if self.guard_rail_generator:
            self._current_guard_rails = self.guard_rail_generator.generate(conflicts)
            batch.guard_rails = self._current_guard_rails
        else:
            # Use SID suggestions as fallback
            for conflict in conflicts:
                self._current_guard_rails.extend(conflict.suggested_guard_rails)
            batch.guard_rails = self._current_guard_rails
        
        return GatingDecision.GATED_TRAINING
    
    def get_guard_rails(self) -> List[str]:
        """Get the current guard-rail statements."""
        return self._current_guard_rails
    
    def get_current_conflicts(self) -> List[SIDResult]:
        """Get the current conflict results."""
        return self._current_conflicts


# ============================================================================
# STEP 5: Guard-Rail Generator (Interface for Phase 2)
# ============================================================================

class GuardRailGenerator(ABC):
    """
    Abstract base class for guard-rail generation.
    
    This is the interface that Phase 2 will implement.
    
    Guard-rails are natural language statements that teach the model
    to learn new facts while respecting existing knowledge.
    
    Example guard-rails for "Penguins can fly" (which conflicts):
        - "Birds generally can fly"
        - "Penguins are birds"
        - "Penguins are an exception - they cannot fly"
    """
    
    @abstractmethod
    def generate(self, conflicts: List[SIDResult]) -> List[str]:
        """
        Generate guard-rail statements for the given conflicts.
        
        Args:
            conflicts: List of SIDResult with conflict information
            
        Returns:
            List of natural language guard-rail statements
        """
        pass
    
    @abstractmethod
    def query_knowledge(self, concept: str) -> Dict[str, Any]:
        """
        Query external knowledge for guard-rail generation.
        
        Args:
            concept: The concept to query (e.g., "penguin")
            
        Returns:
            Dict with general rules, exceptions, related facts
        """
        pass


class BasicGuardRailGenerator(GuardRailGenerator):
    """
    Basic implementation of guard-rail generation.
    
    This provides a working implementation that can be enhanced in Phase 2.
    Uses the hybrid knowledge base for knowledge queries.
    """
    
    def __init__(self, knowledge_base=None):
        """
        Initialize with optional knowledge base.
        
        Args:
            knowledge_base: HybridKnowledgeBase instance (optional)
        """
        self.kb = knowledge_base
        if self.kb is None:
            try:
                from .hybrid_kb import HybridKnowledgeBase
                self.kb = HybridKnowledgeBase()
            except ImportError:
                logger.warning("HybridKnowledgeBase not available")
    
    def generate(self, conflicts: List[SIDResult]) -> List[str]:
        """Generate guard-rail statements from conflicts."""
        guard_rails = []
        
        for conflict in conflicts:
            if not conflict.has_conflict:
                continue
            
            # Extract subject from sample (basic extraction)
            subject = self._extract_subject(conflict.sample)
            
            if subject and self.kb:
                # Query knowledge about the subject
                knowledge = self.query_knowledge(subject)
                
                # Generate guard-rails based on knowledge
                guard_rails.extend(self._create_guard_rails(
                    subject, conflict, knowledge
                ))
            else:
                # Fallback to SID suggestions
                guard_rails.extend(conflict.suggested_guard_rails)
        
        return guard_rails
    
    def query_knowledge(self, concept: str) -> Dict[str, Any]:
        """Query knowledge base for concept information."""
        if not self.kb:
            return {}
        
        result = {
            "categories": [],
            "capabilities": {"can": [], "cannot": []},
            "properties": [],
            "related": []
        }
        
        # Get categories (IsA relations)
        result["categories"] = self.kb.get_categories(concept)
        
        # Get properties
        result["properties"] = self.kb.get_properties(concept)
        
        # Check common capabilities
        for action in ["fly", "swim", "walk", "bark", "meow"]:
            can_do, conf, _ = self.kb.check_capability(concept, action)
            if conf > 0.5:
                if can_do:
                    result["capabilities"]["can"].append(action)
                else:
                    result["capabilities"]["cannot"].append(action)
        
        return result
    
    def _extract_subject(self, text: str) -> Optional[str]:
        """Extract the main subject from text."""
        # Simple extraction - first word or noun phrase
        words = text.lower().replace(".", "").replace(",", "").split()
        if words:
            # Skip articles
            for word in words:
                if word not in ["the", "a", "an", "this", "that"]:
                    return word
        return None
    
    def _create_guard_rails(self, subject: str, conflict: SIDResult,
                            knowledge: Dict[str, Any]) -> List[str]:
        """Create guard-rail statements from knowledge."""
        guard_rails = []
        
        categories = knowledge.get("categories", [])
        cannot_do = knowledge.get("capabilities", {}).get("cannot", [])
        
        # Normalize subject (remove trailing 's' if plural)
        subject_singular = subject.rstrip('s') if subject.endswith('s') and not subject.endswith('ss') else subject
        subject_plural = subject_singular + 's'
        
        # If subject is a subtype of something
        if categories:
            parent = categories[0]
            guard_rails.append(f"{subject_singular.capitalize()} is a type of {parent}.")
            
            # Check if parent can do something subject cannot
            for action in cannot_do:
                parent_can, _, _ = self.kb.check_capability(parent, action) if self.kb else (False, 0, None)
                if parent_can:
                    guard_rails.append(
                        f"While {parent}s generally can {action}, "
                        f"{subject_plural} are an exception and cannot {action}."
                    )
        
        # Add explicit capability statements
        for action in cannot_do:
            guard_rails.append(f"{subject_plural.capitalize()} cannot {action}.")
        
        return guard_rails


# ============================================================================
# STEP 6: SG-CL Trainer Integration Interface
# ============================================================================

class SGCLTrainerInterface(ABC):
    """
    Abstract interface for SG-CL training.
    
    This defines the contract that the Phase 2 trainer must implement.
    """
    
    @abstractmethod
    def normal_train(self, model: Any, batch: TaskBatch) -> Dict[str, float]:
        """
        Normal fine-tuning without guard-rails.
        
        Args:
            model: The LLM to train
            batch: TaskBatch with samples
            
        Returns:
            Training metrics (loss, etc.)
        """
        pass
    
    @abstractmethod
    def gated_train(self, model: Any, batch: TaskBatch,
                    guard_rails: List[str]) -> Dict[str, float]:
        """
        Gated training with guard-rail augmentation.
        
        Args:
            model: The LLM to train
            batch: TaskBatch with samples
            guard_rails: Guard-rail statements to augment training
            
        Returns:
            Training metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint after task."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        pass


# ============================================================================
# STEP 8: Evaluation Interface (SCP Dataset)
# ============================================================================

@dataclass
class SCPProbe:
    """
    A Semantic Consistency Probe for evaluation.
    
    Tests whether the model maintains semantic consistency.
    """
    probe_id: int
    premise: str           # What the model should know
    hypothesis: str        # What we're testing
    expected: bool         # Expected answer (True = consistent)
    category: str = "general"  # Type of probe
    
    def __repr__(self):
        return f"SCPProbe({self.premise} -> {self.hypothesis} = {self.expected})"


class SCPDataset:
    """
    Semantic Consistency Probe Dataset for evaluation.
    
    Used in STEP 8 to evaluate model after each task.
    
    Example:
        >>> scp = SCPDataset()
        >>> scp.add_probe(
        ...     premise="Penguins are birds",
        ...     hypothesis="Penguins can fly",
        ...     expected=False  # Should be False (penguins can't fly)
        ... )
    """
    
    def __init__(self):
        self.probes: List[SCPProbe] = []
        self._probe_counter = 0
    
    def add_probe(self, premise: str, hypothesis: str, expected: bool,
                  category: str = "general") -> SCPProbe:
        """Add a semantic consistency probe."""
        probe = SCPProbe(
            probe_id=self._probe_counter,
            premise=premise,
            hypothesis=hypothesis,
            expected=expected,
            category=category
        )
        self.probes.append(probe)
        self._probe_counter += 1
        return probe
    
    def add_contradiction_probe(self, statement: str, contradiction: str) -> SCPProbe:
        """Add a probe testing for contradiction detection."""
        return self.add_probe(
            premise=statement,
            hypothesis=contradiction,
            expected=False,
            category="contradiction"
        )
    
    def add_consistency_probe(self, statement: str, inference: str) -> SCPProbe:
        """Add a probe testing for consistent inference."""
        return self.add_probe(
            premise=statement,
            hypothesis=inference,
            expected=True,
            category="consistency"
        )
    
    def __iter__(self) -> Iterator[SCPProbe]:
        return iter(self.probes)
    
    def __len__(self) -> int:
        return len(self.probes)
    
    def get_by_category(self, category: str) -> List[SCPProbe]:
        """Get probes by category."""
        return [p for p in self.probes if p.category == category]


@dataclass
class EvaluationMetrics:
    """
    Evaluation metrics for SG-CL.
    
    These metrics measure semantic stability during continual learning.
    """
    task_id: int
    
    # Core metrics
    forgetting_score: float = 0.0        # How much old knowledge is forgotten
    semantic_consistency: float = 0.0     # Consistency with known facts
    contradiction_rate: float = 0.0       # Rate of contradictory outputs
    knowledge_integrity: float = 0.0      # Overall knowledge preservation
    
    # Detailed breakdown
    probes_tested: int = 0
    probes_passed: int = 0
    probes_failed: int = 0
    
    # Per-category metrics
    category_scores: Dict[str, float] = field(default_factory=dict)
    
    def accuracy(self) -> float:
        """Overall accuracy on probes."""
        if self.probes_tested == 0:
            return 0.0
        return self.probes_passed / self.probes_tested
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "forgetting_score": self.forgetting_score,
            "semantic_consistency": self.semantic_consistency,
            "contradiction_rate": self.contradiction_rate,
            "knowledge_integrity": self.knowledge_integrity,
            "accuracy": self.accuracy(),
            "probes_tested": self.probes_tested,
            "probes_passed": self.probes_passed,
            "category_scores": self.category_scores
        }


class SCPEvaluator(ABC):
    """
    Abstract evaluator interface for Phase 2.
    
    Evaluates the model using SCP probes after each task.
    """
    
    @abstractmethod
    def evaluate(self, model: Any, scp_dataset: SCPDataset,
                 task_id: int) -> EvaluationMetrics:
        """
        Evaluate model on SCP probes.
        
        Args:
            model: The trained model
            scp_dataset: SCP probes to test
            task_id: Current task ID
            
        Returns:
            EvaluationMetrics with all scores
        """
        pass


# ============================================================================
# FULL PIPELINE ORCHESTRATOR
# ============================================================================

class SGCLPipeline:
    """
    Full SG-CL Pipeline Orchestrator.
    
    Coordinates all components of the SG-CL system:
    1. SeCA Dataset (sequential tasks)
    2. SID (conflict detection)
    3. Gating Controller (decision point)
    4. Guard-Rail Generator
    5. SG-CL Trainer
    6. SCP Evaluator
    
    Example:
        >>> from sid import create_detector
        >>> from sid.pipeline import SGCLPipeline, SeCADataset
        >>> 
        >>> # Create pipeline
        >>> detector = create_detector(offline_only=True)
        >>> pipeline = SGCLPipeline(detector)
        >>> 
        >>> # Add tasks
        >>> pipeline.dataset.add_task("Birds can fly", "Sparrows are birds")
        >>> pipeline.dataset.add_task("Penguins are birds", "Penguins cannot fly")
        >>> 
        >>> # Run pipeline (simulation mode - no actual LLM training)
        >>> results = pipeline.run_simulation()
    """
    
    def __init__(self, detector, 
                 trainer: Optional[SGCLTrainerInterface] = None,
                 evaluator: Optional[SCPEvaluator] = None,
                 guard_rail_generator: Optional[GuardRailGenerator] = None):
        """
        Initialize the SG-CL pipeline.
        
        Args:
            detector: SemanticInconsistencyDetector instance
            trainer: SGCLTrainerInterface implementation (Phase 2)
            evaluator: SCPEvaluator implementation (Phase 2)
            guard_rail_generator: GuardRailGenerator (basic provided)
        """
        # Core components
        self.sid_adapter = SIDPipelineAdapter(detector)
        self.gating_controller = GatingController(
            self.sid_adapter,
            guard_rail_generator or BasicGuardRailGenerator()
        )
        
        # Optional Phase 2 components
        self.trainer = trainer
        self.evaluator = evaluator
        
        # Datasets
        self.dataset = SeCADataset()
        self.scp_dataset = SCPDataset()
        
        # State
        self._task_history: List[Dict[str, Any]] = []
        self._evaluation_history: List[EvaluationMetrics] = []
    
    def add_task(self, *samples: str, name: Optional[str] = None) -> Task:
        """Add a task to the dataset."""
        return self.dataset.add_task(*samples, name=name)
    
    def add_scp_probe(self, premise: str, hypothesis: str, 
                      expected: bool) -> SCPProbe:
        """Add an SCP evaluation probe."""
        return self.scp_dataset.add_probe(premise, hypothesis, expected)
    
    def run_simulation(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the pipeline in simulation mode (no actual LLM training).
        
        This demonstrates the full pipeline flow and can be used for testing.
        
        Returns:
            Dict with pipeline execution results
        """
        results = {
            "tasks_processed": 0,
            "samples_analyzed": 0,
            "conflicts_detected": 0,
            "gated_batches": 0,
            "normal_batches": 0,
            "guard_rails_generated": 0,
            "task_details": []
        }
        
        if verbose:
            print("=" * 60)
            print("SG-CL PIPELINE SIMULATION")
            print("=" * 60)
            print()
        
        # Process each task
        for task in self.dataset:
            if verbose:
                print(f"--- TASK {task.task_id}: {task.name} ---")
                print(f"Samples: {len(task)}")
            
            task_result = {
                "task_id": task.task_id,
                "task_name": task.name,
                "samples": len(task),
                "conflicts": [],
                "gating_decisions": [],
                "guard_rails": []
            }
            
            # Process batches
            for batch in self.dataset.get_batches(task, batch_size=4):
                results["samples_analyzed"] += len(batch.samples)
                
                # Step 3 & 4: SID analysis and gating decision
                decision = self.gating_controller.process_batch(batch)
                task_result["gating_decisions"].append(decision.value)
                
                if decision == GatingDecision.GATED_TRAINING:
                    results["gated_batches"] += 1
                    results["conflicts_detected"] += len(batch.conflicts)
                    results["guard_rails_generated"] += len(batch.guard_rails)
                    
                    task_result["conflicts"].extend(
                        [c.sample for c in batch.conflicts]
                    )
                    task_result["guard_rails"].extend(batch.guard_rails)
                    
                    if verbose:
                        for conflict in batch.conflicts:
                            print(f"  [CONFLICT] \"{conflict.sample}\"")
                            print(f"             -> {conflict.conflict_type}")
                        print(f"  [GUARD-RAILS] {batch.guard_rails[:2]}...")
                else:
                    results["normal_batches"] += 1
                    if verbose:
                        print(f"  [OK] Batch {batch.batch_idx}: Normal training")
            
            results["task_details"].append(task_result)
            results["tasks_processed"] += 1
            
            if verbose:
                print()
        
        # Summary
        if verbose:
            print("=" * 60)
            print("PIPELINE SUMMARY")
            print("=" * 60)
            print(f"Tasks processed: {results['tasks_processed']}")
            print(f"Samples analyzed: {results['samples_analyzed']}")
            print(f"Conflicts detected: {results['conflicts_detected']}")
            print(f"Normal batches: {results['normal_batches']}")
            print(f"Gated batches: {results['gated_batches']}")
            print(f"Guard-rails generated: {results['guard_rails_generated']}")
            print()
        
        return results
    
    def run(self, model: Any, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the full pipeline with actual training.
        
        Requires trainer and evaluator to be set (Phase 2).
        
        Args:
            model: The LLM to train
            verbose: Print progress
            
        Returns:
            Dict with training results and evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError(
                "Trainer not set. Use run_simulation() for testing, "
                "or set a trainer for actual training."
            )
        
        results = {
            "tasks_processed": 0,
            "training_metrics": [],
            "evaluation_metrics": []
        }
        
        for task in self.dataset:
            if verbose:
                print(f"Training on Task {task.task_id}: {task.name}")
            
            # Process each batch
            for batch in self.dataset.get_batches(task):
                decision = self.gating_controller.process_batch(batch)
                
                if decision == GatingDecision.NORMAL_TRAINING:
                    metrics = self.trainer.normal_train(model, batch)
                else:
                    guard_rails = self.gating_controller.get_guard_rails()
                    metrics = self.trainer.gated_train(model, batch, guard_rails)
                
                results["training_metrics"].append(metrics)
            
            # Save checkpoint after task
            self.trainer.save_checkpoint(f"checkpoint_task_{task.task_id}")
            
            # Evaluate after task
            if self.evaluator:
                eval_metrics = self.evaluator.evaluate(
                    model, self.scp_dataset, task.task_id
                )
                results["evaluation_metrics"].append(eval_metrics.to_dict())
                
                if verbose:
                    print(f"  Accuracy: {eval_metrics.accuracy():.2%}")
                    print(f"  Semantic Consistency: {eval_metrics.semantic_consistency:.2%}")
            
            results["tasks_processed"] += 1
        
        return results
    
    def get_sid_statistics(self) -> Dict[str, Any]:
        """Get SID analysis statistics."""
        return self.sid_adapter.get_statistics()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_pipeline(offline_only: bool = True) -> SGCLPipeline:
    """
    Create a configured SG-CL pipeline.
    
    Args:
        offline_only: Use offline knowledge only (no API calls)
        
    Returns:
        Configured SGCLPipeline instance
    """
    from .detector import create_detector
    
    detector = create_detector(offline_only=offline_only)
    return SGCLPipeline(detector)


def demo_pipeline():
    """Demonstrate the SG-CL pipeline."""
    print("Creating SG-CL Pipeline...")
    pipeline = create_pipeline(offline_only=True)
    
    # Add SeCA tasks
    pipeline.add_task(
        "Birds can fly",
        "Sparrows are birds",
        "Eagles are birds",
        name="General Bird Knowledge"
    )
    
    pipeline.add_task(
        "Penguins are birds",
        "Penguins can fly",  # This conflicts!
        "Penguins live in Antarctica",
        name="Penguin Knowledge"
    )
    
    pipeline.add_task(
        "Ostriches are birds",
        "Ostriches cannot fly",
        "Ostriches can run fast",
        name="Ostrich Knowledge"
    )
    
    # Add SCP probes
    pipeline.add_scp_probe(
        premise="Penguins are birds",
        hypothesis="Penguins can fly",
        expected=False
    )
    
    pipeline.add_scp_probe(
        premise="Birds can fly",
        hypothesis="Sparrows can fly",
        expected=True
    )
    
    # Run simulation
    results = pipeline.run_simulation(verbose=True)
    
    return results


if __name__ == "__main__":
    demo_pipeline()

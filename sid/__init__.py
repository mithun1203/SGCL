"""
Semantic Inconsistency Detector (SID) Module
=============================================

A neuro-symbolic module for detecting semantic conflicts in knowledge statements.
Part of the Symbolic-Gated Continual Learning (SG-CL) Framework.

Author: Mithun Naik
Project: SGCL - Capstone Project
Version: 1.0.0

Main Components:
    - SemanticInconsistencyDetector: Main detector class
    - ConceptNetClient: Interface to ConceptNet knowledge base
    - HybridKnowledgeBase: Offline-first KB with embeddings fallback
    - SGCLPipeline: Full pipeline orchestrator for integration
    - EntityExtractor: NLP-based entity extraction
    - RelationMapper: Maps natural language to semantic relations
    - ConflictEngine: Core conflict detection logic

Pipeline Integration:
    The SID module is designed for seamless integration into the SG-CL pipeline:
    
    1. SeCA Dataset -> Sequential task input
    2. SID Analysis -> Conflict detection
    3. Gating Controller -> Training path decision
    4. Guard-Rail Generator -> Augmentation for conflicts
    5. SG-CL Trainer -> Normal or gated training
    6. SCP Evaluator -> Semantic consistency evaluation

Usage:
    # Basic conflict detection
    from sid import create_detector
    detector = create_detector(offline_only=True)
    result = detector.detect_conflict("Penguins can fly")
    
    # Full pipeline simulation
    from sid.pipeline import create_pipeline
    pipeline = create_pipeline()
    pipeline.add_task("Birds can fly", "Penguins are birds")
    pipeline.add_task("Penguins can fly")  # Conflicts!
    results = pipeline.run_simulation()
"""

from .detector import SemanticInconsistencyDetector, create_detector, SIDConfig
from .conceptnet_client import ConceptNetClient
from .entity_extractor import EntityExtractor
from .relation_mapper import RelationMapper
from .conflict_engine import ConflictEngine
from .models import (
    Triple,
    ConflictResult,
    ExtractionResult,
    ConceptNetEdge,
    SemanticRelation
)

# Import Numberbatch/Hybrid KB (optional - may not be configured)
try:
    from .numberbatch_kb import NumberbatchKB, NumberbatchConfig, create_numberbatch_kb
    from .hybrid_kb import HybridKnowledgeBase, HybridKBConfig, create_hybrid_kb
    HAS_ADVANCED_KB = True
except ImportError:
    HAS_ADVANCED_KB = False

# Import SeCA Dataset components
try:
    from .seca_dataset import (
        SeCADataset as RichSeCADataset,  # Full-featured dataset
        Sample as SeCASample,
        Task as SeCATask,
        ConflictType as SeCAConflictType,
        create_seca_dataset,
        print_dataset_summary as print_seca_summary
    )
    HAS_SECA = True
except ImportError:
    HAS_SECA = False

# Import Pipeline components
try:
    from .pipeline import (
        SGCLPipeline,
        create_pipeline,
        SeCADataset,  # Simple pipeline wrapper
        SCPDataset,
        SIDPipelineAdapter,
        GatingController,
        GatingDecision,
        BasicGuardRailGenerator,
        GuardRailGenerator,
        SGCLTrainerInterface,
        SCPEvaluator,
        EvaluationMetrics,
        Task,
        TaskBatch,
        SCPProbe,
        SIDResult
    )
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False

__version__ = "1.0.0"
__author__ = "Mithun Naik"
__project__ = "Symbolic-Gated Continual Learning (SG-CL)"

__all__ = [
    # Core components
    "SemanticInconsistencyDetector",
    "create_detector",
    "SIDConfig",
    "ConceptNetClient",
    "EntityExtractor",
    "RelationMapper",
    "ConflictEngine",
    
    # Data models
    "Triple",
    "ConflictResult",
    "ExtractionResult",
    "ConceptNetEdge",
    "SemanticRelation",
    
    # Advanced KB (Numberbatch/Hybrid)
    "NumberbatchKB",
    "NumberbatchConfig",
    "create_numberbatch_kb",
    "HybridKnowledgeBase",
    "HybridKBConfig", 
    "create_hybrid_kb",
    "HAS_ADVANCED_KB",
    
    # SeCA Dataset (Rich version)
    "RichSeCADataset",
    "SeCASample",
    "SeCATask",
    "SeCAConflictType",
    "create_seca_dataset",
    "print_seca_summary",
    "HAS_SECA",
    
    # Pipeline components
    "SGCLPipeline",
    "create_pipeline",
    "SeCADataset",  # Simple pipeline wrapper
    "SCPDataset",
    "SIDPipelineAdapter",
    "GatingController",
    "GatingDecision",
    "BasicGuardRailGenerator",
    "GuardRailGenerator",
    "SGCLTrainerInterface",
    "SCPEvaluator",
    "EvaluationMetrics",
    "Task",
    "TaskBatch",
    "SCPProbe",
    "SIDResult",
    "HAS_PIPELINE",
]

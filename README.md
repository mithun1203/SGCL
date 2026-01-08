# Symbolic-Gated Continual Learning (SG-CL)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A neuro-symbolic framework for continual learning in Large Language Models with conflict detection and guardrails.

## 📁 Project Structure

```
SGCL/
├── sid/                    # Semantic Inconsistency Detector (Core)
│   ├── detector.py         # Main conflict detector
│   ├── pipeline.py         # Complete SID pipeline
│   ├── conflict_engine.py  # Conflict detection logic
│   ├── entity_extractor.py # NLP entity extraction
│   ├── relation_mapper.py  # Relation mapping
│   ├── hybrid_kb.py        # Knowledge base interface
│   ├── conceptnet_client.py# ConceptNet integration
│   ├── knowledge_base.json # Mini KB (offline)
│   ├── knowledge_base_full.json # Full ConceptNet (142K concepts)
│   └── seca_10k_dataset.json   # 10K training dataset
│
├── guardrail/              # SG-CL Guardrails
│   ├── guardrail_controller.py # Training guardrail controller
│   └── guardrail_generator.py  # Guardrail generation
│
├── seca/                   # SeCA Dataset Tools
│   ├── view_seca_dataset.py    # Dataset viewer
│   ├── generate_augmented_dataset.py # Dataset generator
│   ├── audit_and_fix_dataset.py     # Dataset validator
│   └── evaluation_splits/  # Train/test splits
│
├── scp/                    # SCP Evaluation
│   └── scp_evaluation.py   # Semantic Consistency metrics
│
├── scripts/                # Utility Scripts
│   ├── run_quick_test.py   # Quick validation test
│   ├── run_mini_cpu_experiment.py # CPU-only test
│   ├── verify_integration.py      # Integration tests
│   ├── download_full_conceptnet.py # KB downloader
│   └── upgrade_conceptnet_kb.py    # KB upgrade script
│
├── docs/                   # Documentation
│   ├── SECA_10K_FINAL.md   # Dataset documentation
│   ├── COMPLETE_SYSTEM.md  # System overview
│   ├── KAGGLE_SETUP.md     # Kaggle instructions
│   └── ...
│
├── Core Files
│   ├── sgcl_training.py    # SG-CL training engine
│   ├── sgcl_data_loader.py # SeCA data loader
│   ├── baseline_methods.py # Baseline methods (Naive, EWC, Replay)
│   ├── results_analysis.py # Visualization & analysis
│   ├── run_full_experiments.py    # Experiment orchestrator
│   └── kaggle_sgcl_final.ipynb   # Kaggle notebook
│
├── tests/                  # Unit Tests
└── requirements.txt        # Python dependencies
```

---

## 🎯 What Problem Does SID Solve?

During sequential fine-tuning of LLMs, models can learn contradictory information:

```
Task 1: "All birds can fly"
Task 2: "Penguins are birds"  
Result: Model incorrectly infers "Penguins can fly" ❌
```

**SID detects such semantic conflicts BEFORE training by:**
1. Extracting semantic triples from input text
2. Querying external knowledge (ConceptNet) for related facts
3. Applying conflict detection rules
4. Performing inheritance-based reasoning

---

## ✨ Key Features

- ✅ **Multi-backend NLP**: Supports spaCy, Stanza, and rule-based extraction
- ✅ **ConceptNet Integration**: Queries commonsense knowledge for conflict detection
- ✅ **Offline Mode**: Works without internet using pre-loaded knowledge base
- ✅ **Caching**: Efficient caching of knowledge base queries
- ✅ **Inheritance Reasoning**: Detects conflicts through type hierarchies
- ✅ **Detailed Evidence**: Provides reasoning chains for detected conflicts
- ✅ **Batch Processing**: Efficiently process multiple statements
- ✅ **Easily Integrable**: Clean API for SG-CL training loop integration

---

## 🏗️ Architecture

```
SID Module Architecture
═══════════════════════

┌─────────────────────────────────────────────────────────────┐
│                    Input Text                                │
│              "Penguins can fly"                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Entity Extractor                            │
│    (spaCy / Stanza / Rule-based)                            │
│    Entities: [penguin, fly]                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Relation Mapper                             │
│    Pattern matching + Dependency parsing                     │
│    Triple: (penguin, CapableOf, fly)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                ConceptNet Client                             │
│    KB Facts: (penguin, NotCapableOf, fly)                   │
│              (penguin, IsA, bird)                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Conflict Engine                             │
│    Result: CONFLICT DETECTED                                 │
└─────────────────────────────────────────────────────────────┘
```

**Project Structure:**
```
SGCL new/
├── sid/
│   ├── __init__.py              # Module exports
│   ├── models.py                # Data models (Triple, ConflictResult, etc.)
│   ├── detector.py              # Main SemanticInconsistencyDetector
│   ├── conceptnet_client.py     # ConceptNet API client
│   ├── entity_extractor.py      # NLP entity extraction
│   ├── relation_mapper.py       # Text to relation mapping
│   ├── conflict_engine.py       # Conflict detection logic
│   ├── hybrid_kb.py             # Hybrid knowledge base (JSON + embeddings)
│   ├── numberbatch_kb.py        # ConceptNet Numberbatch embeddings
│   ├── download_numberbatch.py  # Download script for mini embeddings
│   ├── knowledge_base.json      # Offline knowledge (54+ concepts)
│   └── data/
│       └── mini.h5              # Numberbatch embeddings (optional, ~150MB)
├── tests/
│   ├── __init__.py
│   └── test_sid.py              # Comprehensive test suite (100+ tests)
├── examples/
│   └── usage_examples.py        # Usage demonstrations
├── demo.py                      # Quick demo script
├── test_hybrid_kb.py            # Hybrid KB test suite
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 🔄 SG-CL Pipeline Integration

SID is designed for seamless integration into the full SG-CL pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SG-CL PIPELINE FLOW                              │
└─────────────────────────────────────────────────────────────────────┘

  SeCA Dataset          SID Analysis         Gating Decision
  ┌─────────┐          ┌───────────┐         ┌───────────┐
  │ Task 1  │────────▶ │  Conflict │────────▶│  Normal   │───▶ LLM Train
  │ Task 2  │          │  Check    │         │   or      │
  │ Task 3  │          └───────────┘         │  Gated?   │
  └─────────┘                │               └───────────┘
                             │                     │
                             ▼                     ▼
                      ┌───────────┐         ┌───────────┐
                      │ Guard-Rail│◀────────│  Gated    │
                      │ Generator │         │ Training  │
                      └───────────┘         └───────────┘
                                                   │
                                                   ▼
                                            ┌───────────┐
                                            │    SCP    │
                                            │ Evaluation│
                                            └───────────┘
```

### Pipeline Components

| Component | Status | Description |
|-----------|--------|-------------|
| `SeCADataset` | ✅ Ready | Sequential task input format |
| `SIDPipelineAdapter` | ✅ Ready | SID integration for pipeline |
| `GatingController` | ✅ Ready | Training path decision |
| `BasicGuardRailGenerator` | ✅ Ready | Guard-rail generation |
| `SGCLTrainerInterface` | 📋 Interface | LLM training (Phase 2) |
| `SCPEvaluator` | 📋 Interface | Evaluation (Phase 2) |

### Pipeline Usage

```python
from sid.pipeline import create_pipeline

# Create pipeline
pipeline = create_pipeline(offline_only=True)

# Add sequential tasks (SeCA format)
pipeline.add_task("Birds can fly", "Sparrows are birds")
pipeline.add_task("Penguins are birds", "Penguins can fly")  # Conflict!

# Add evaluation probes
pipeline.add_scp_probe(
    premise="Penguins are birds",
    hypothesis="Penguins can fly",
    expected=False
)

# Run simulation (no LLM needed)
results = pipeline.run_simulation()

# Output:
# Conflicts detected: 1
# Guard-rails generated: ["Penguin is a type of bird.", 
#                         "While birds can fly, penguins are an exception..."]
```

### Pipeline Demo Output

```
============================================================
SG-CL PIPELINE SIMULATION
============================================================

--- TASK 0: General Bird Knowledge ---
  [OK] Batch 0: Normal training

--- TASK 1: Penguin Knowledge ---
  [CONFLICT] "Penguins can fly."
             -> direct_contradiction
  [GUARD-RAILS] ['Penguin is a type of bird.', 
                 'While birds generally can fly, penguins are an exception...']

============================================================
PIPELINE SUMMARY
============================================================
Tasks processed: 3
Conflicts detected: 1
Guard-rails generated: 3
```

---

## 📊 SeCA Dataset

**SeCA (Semantic Consistency Aware) v2.0 - 10K Edition**

The benchmark dataset for evaluating semantic conflict detection and continual learning.

### Dataset Composition

SeCA consists of approximately 10,000 samples distributed across sequential tasks. The dataset is anchored by 320 manually curated samples and expanded via controlled paraphrasing and template-based augmentation. All augmented samples are validated using SID to preserve semantic correctness.

**Statistics**:
- **Total Samples**: 10,000 across 16 tasks
- **Per Task**: 625 samples (500 train / 125 test)
- **Conflict Rate**: 48.6% overall
- **Core Quality**: 320 manually curated samples
- **Augmented**: 9,680 systematically generated samples

### Limitations

While a portion of SeCA is synthetically augmented, this design enables controlled analysis of semantic conflict and continual learning behavior. Future work will expand SeCA with more naturally occurring text.

### Conflict Types

- **Direct Contradiction** (16.1%): Statements that directly contradict each other
- **Exception Violation** (15.7%): General rules with missing exceptions
- **Attribute Conflict** (15.7%): Incompatible property assignments
- **Delayed Conflict** (0.4%): Conflicts emerging across tasks
- **Paraphrase Conflict** (0.4%): Semantically equivalent conflicts

See `docs/SECA_10K_FINAL.md` for complete documentation.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/sgcl.git
cd sgcl

# Install dependencies
pip install requests

# Optional: Install NLP backends for better extraction
pip install spacy
python -m spacy download en_core_web_sm

# Optional: Install Numberbatch embeddings support
pip install h5py numpy
```

### Knowledge Sources

SID supports multiple knowledge sources for flexibility:

| Source | Size | Description | Use Case |
|--------|------|-------------|----------|
| **JSON KB** | ~50 KB | Curated facts for 54+ concepts | Default, always available |
| **ConceptNet API** | Online | Live API queries | Full coverage (when available) |
| **Numberbatch Mini** | ~150 MB | Semantic embeddings | Offline inference for unknown concepts |

#### "ConceptNet Mini" Mode (Recommended for Offline Use)

For reliable offline operation without the 9GB full ConceptNet database:

```bash
# Download Numberbatch Mini embeddings (~150MB)
python -m sid.download_numberbatch --type mini
```

This provides:
- ✅ Semantic similarity for ~500K concepts
- ✅ Works completely offline
- ✅ Fast inference (<10ms per query)
- ✅ Only ~150MB storage (vs 9GB for full database)

### Basic Usage

```python
from sid import SemanticInconsistencyDetector, create_detector

# Create detector (uses offline mode by default)
detector = create_detector()

# Check for conflicts
result = detector.detect_conflict("Penguins can fly")

print(f"Has conflict: {result.has_conflict}")  # True
print(f"Conflicts: {result.conflict_count}")   # 1

# Get detailed explanation
if result.has_conflict:
    for conflict in result.conflicts:
        print(conflict.explain())
```

### Using the Hybrid Knowledge Base Directly

```python
from sid.hybrid_kb import HybridKnowledgeBase

# Create hybrid KB (JSON + Numberbatch)
kb = HybridKnowledgeBase()

# Check capability
can_fly, confidence, source = kb.check_capability("penguin", "fly")
print(f"Penguin can fly: {can_fly}")  # False
print(f"Confidence: {confidence}")    # 1.0
print(f"Source: {source}")            # json_kb

# Check relationships
is_bird, conf, src = kb.check_relationship("penguin", "is_a", "bird")
print(f"Penguin is a bird: {is_bird}")  # True
```

### Batch Processing

```python
texts = [
    "Birds can fly",
    "Penguins are birds", 
    "Penguins can fly"  # This conflicts!
]

batch_result = detector.detect_conflicts_batch(texts)
print(batch_result.summary())
```

### SG-CL Training Loop Integration

```python
from sid import create_detector

detector = create_detector()

def sgcl_training_loop(model, tasks):
    for task in tasks:
        for batch in task:
            for sample in batch:
                # SID conflict detection
                result = detector.detect_conflict(sample.text)
                
                if result.has_conflict:
                    # Apply gated training + guardrails
                    sample.requires_gating = True
                    gated_train(model, batch, result.conflicts)
                else:
                    # Normal fine-tuning
                    normal_train(model, batch)
```

---

## 📚 API Reference

### SemanticInconsistencyDetector

| Method | Description |
|--------|-------------|
| `detect_conflict(text)` | Detect conflicts in text, returns `ConflictResult` |
| `detect_conflicts_batch(texts)` | Process multiple texts |
| `is_conflicting(text)` | Quick boolean check |
| `get_conflicts(text)` | Get list of conflicts |
| `check_triple(triple)` | Check a pre-extracted triple |
| `extract_triples(text)` | Extract triples without conflict checking |
| `query_knowledge(concept)` | Query knowledge base |
| `explain(text)` | Get detailed analysis |

### ConflictResult

| Property | Description |
|----------|-------------|
| `has_conflict` | Boolean - whether conflicts detected |
| `conflict_count` | Number of conflicts |
| `conflicts` | List of ConflictEvidence |
| `extracted_triples` | Triples from input |
| `processing_time` | Time in seconds |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sid --cov-report=html

# Run specific tests
pytest tests/test_sid.py::TestSemanticInconsistencyDetector -v
```

### Test Coverage

- `TestTriple`: Triple data structure tests
- `TestConceptNetClient`: Knowledge base client tests
- `TestEntityExtractor`: NLP extraction tests
- `TestRelationMapper`: Relation mapping tests
- `TestConflictEngine`: Conflict detection tests
- `TestSemanticInconsistencyDetector`: Main detector tests
- `TestIntegration`: End-to-end tests
- `TestEdgeCases`: Edge case handling

---

## 📊 Performance

- **Detection time**: <100ms per sentence (with caching)
- **Cache hit rate**: >90% for common concepts
- **Offline KB**: 54+ pre-loaded concepts for instant lookup
- **Numberbatch**: ~500K concepts via semantic similarity
- **Batch processing**: Efficient parallel knowledge retrieval
- **Storage**: Only ~150MB for full offline operation

### Test Results

```
============================================================
HYBRID KNOWLEDGE BASE TEST SUITE
============================================================

Total Passed: 39
Total Failed: 0
Success Rate: 100.0%

==> ALL TESTS PASSED!
```

---

## 🔮 Phase 2 Roadmap

- [ ] Guard-rail Generator implementation
- [ ] SG-CL Trainer with gated updates
- [ ] SeCA dataset creation
- [ ] SCP evaluation metrics
- [ ] Full LLM training experiments

---

## 📝 Citation

```bibtex
@misc{sgcl2024,
  title={Symbolic-Gated Continual Learning: A Neuro-Symbolic Framework 
         for Mitigating Semantic Inconsistency in Generative Models},
  author={Mithun Naik},
  year={2024},
  note={Capstone Project}
}
```

---

## 📄 License

MIT License

---

**Author**: Mithun Naik  
**Project**: SGCL Capstone  
**Version**: 1.0.0

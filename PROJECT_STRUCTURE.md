# SG-CL Project Organization

## Directory Structure

```
SGCL/
│
├── 📁 sid/                          # Semantic Inconsistency Detector (Core Component)
│   ├── detector.py                  # Main SID detector class
│   ├── pipeline.py                  # Complete detection pipeline
│   ├── conflict_engine.py           # Conflict detection logic & rules
│   ├── entity_extractor.py          # NLP-based entity extraction (spaCy/Stanza)
│   ├── relation_mapper.py           # ConceptNet relation mapping
│   ├── hybrid_kb.py                 # Hybrid knowledge base (offline + online)
│   ├── conceptnet_client.py         # ConceptNet API client
│   ├── models.py                    # Data models (Triple, ConflictResult)
│   ├── knowledge_base.json          # Mini KB (3.5K concepts, offline)
│   ├── knowledge_base_full.json     # Full KB (142K concepts, 55 MB)
│   └── seca_10k_dataset.json        # SeCA v2.0 - 10K samples dataset
│
├── 📁 guardrail/                    # SG-CL Guardrail System
│   ├── guardrail_controller.py      # Main controller for training integration
│   └── guardrail_generator.py       # Dynamic guardrail generation
│
├── 📁 seca/                         # SeCA Dataset Tools
│   ├── view_seca_dataset.py         # Dataset viewer & statistics
│   ├── generate_augmented_dataset.py # Scale dataset (320 → 10K)
│   ├── audit_and_fix_dataset.py     # Dataset validation & repair
│   └── evaluation_splits/           # Train/test split data
│
├── 📁 scp/                          # Semantic Consistency Preservation (SCP)
│   └── scp_evaluation.py            # SCP metrics evaluation
│       - Semantic Consistency Score
│       - Contradiction Rate
│       - Forgetting Rate
│
├── 📁 scripts/                      # Utility & Testing Scripts
│   ├── run_quick_test.py            # Quick validation test
│   ├── run_mini_cpu_experiment.py   # CPU-only mini experiment
│   ├── verify_integration.py        # Integration verification
│   ├── download_full_conceptnet.py  # Download full ConceptNet KB
│   └── upgrade_conceptnet_kb.py     # Upgrade KB version
│
├── 📁 docs/                         # Documentation
│   ├── SECA_10K_FINAL.md            # 10K dataset documentation
│   ├── COMPLETE_SYSTEM.md           # Complete system overview
│   ├── KAGGLE_SETUP.md              # Kaggle deployment guide
│   ├── INSTALLATION.md              # Installation instructions
│   ├── GUARDRAIL_SUMMARY.md         # Guardrail documentation
│   ├── TRAINING_INTEGRATION.md      # Training integration guide
│   └── ...                          # Additional documentation
│
├── 📁 tests/                        # Unit Tests
│   ├── test_sid.py                  # SID component tests
│   └── test_seca_dataset.py         # Dataset tests
│
├── 📁 data/                         # Raw Data
│   └── conceptnet-assertions-5.7.0.csv.gz  # ConceptNet raw data
│
├── 📁 experiments/                  # Experiment Results
│   └── full_experiment_YYYYMMDD_HHMMSS/    # Timestamped results
│
├── 📄 Core Training Files
│   ├── sgcl_training.py             # SG-CL training engine (main)
│   ├── sgcl_data_loader.py          # SeCA dataset loader
│   ├── baseline_methods.py          # Baseline methods
│   │   - Naive Fine-tuning
│   │   - EWC (Elastic Weight Consolidation)
│   │   - Experience Replay
│   ├── results_analysis.py          # Visualization & analysis
│   └── run_full_experiments.py      # Experiment orchestrator
│
├── 📄 Deployment
│   └── kaggle_sgcl_final.ipynb      # Kaggle notebook (optimized)
│
└── 📄 Configuration
    ├── README.md                    # Project README
    ├── requirements.txt             # Python dependencies
    └── .gitignore                   # Git ignore rules
```

---

## Component Details

### 🎯 SID (Semantic Inconsistency Detector)
**Purpose**: Detect semantic conflicts in training data before fine-tuning  
**Location**: `sid/`  
**Key Files**:
- `detector.py` - Main detector with batch processing
- `pipeline.py` - End-to-end detection pipeline
- `conflict_engine.py` - Rule-based conflict detection logic
- `hybrid_kb.py` - Knowledge base with offline/online modes

### 🛡️ Guardrails
**Purpose**: Dynamic conflict prevention during training  
**Location**: `guardrail/`  
**Key Files**:
- `guardrail_controller.py` - Integrates with training loop
- `guardrail_generator.py` - Generates task-specific guardrails

### 📊 SeCA Dataset
**Purpose**: Sequential Conflict Awareness dataset for evaluation  
**Location**: `seca/` + `sid/seca_10k_dataset.json`  
**Size**: 10,000 samples across 16 tasks  
**Structure**: 
- 320 manually curated core samples
- 9,680 augmented samples
- 48.2% conflict rate
- Full semantic annotations

### 📈 SCP Evaluation
**Purpose**: Measure semantic consistency preservation  
**Location**: `scp/`  
**Metrics**:
- **Consistency Score**: How well the model maintains learned facts
- **Contradiction Rate**: Percentage of conflicting predictions
- **Forgetting Rate**: Knowledge retention across tasks

### 🔬 Training System
**Purpose**: Complete continual learning pipeline  
**Key Files**:
- `sgcl_training.py` - SG-CL with conflict detection + guardrails
- `baseline_methods.py` - Naive, EWC, Experience Replay
- `run_full_experiments.py` - Orchestrates all experiments
- `results_analysis.py` - Generates plots, tables, LaTeX

---

## Quick Start

### 1. Run Quick Test (CPU, 2 minutes)
```bash
python scripts/run_quick_test.py
```

### 2. Run Full Experiments (GPU, 4 hours)
```bash
python run_full_experiments.py
```

### 3. View Dataset
```bash
python seca/view_seca_dataset.py
```

### 4. Run on Kaggle
Upload `kaggle_sgcl_final.ipynb` to Kaggle with GPU enabled.

---

## File Organization Principles

### ✅ Organized Structure
- **Modular**: Each component in its own directory
- **Discoverable**: Clear naming and grouping
- **Minimal**: No duplicates or unnecessary files
- **Documented**: Each directory has a clear purpose

### ❌ Old Issues (Fixed)
- ~~Duplicate demo files scattered in root~~
- ~~Multiple dataset versions~~
- ~~Mixed documentation files~~
- ~~Test files outside `tests/`~~
- ~~Unclear project structure~~

---

## Import Conventions

When importing from organized directories:

```python
# SID components
from sid.detector import SemanticInconsistencyDetector
from sid.pipeline import SIDPipeline

# Guardrails
from guardrail.guardrail_controller import GuardrailController

# SCP evaluation
from scp.scp_evaluation import compare_methods

# Core training
from sgcl_training import SGCLTrainer
from baseline_methods import NaiveFinetuningTrainer
```

---

## Documentation Map

| Document | Purpose | Location |
|----------|---------|----------|
| README.md | Project overview | Root |
| PROJECT_STRUCTURE.md | This file | Root |
| SECA_10K_FINAL.md | Dataset documentation | `docs/` |
| COMPLETE_SYSTEM.md | System architecture | `docs/` |
| KAGGLE_SETUP.md | Deployment guide | `docs/` |
| INSTALLATION.md | Setup instructions | `docs/` |

---

**Last Updated**: January 8, 2026  
**Version**: 2.0 (Organized)

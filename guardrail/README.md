# Symbolic Guardrail System

**Training-time data augmentation for semantic consistency in continual learning**

## Overview

The Symbolic Guardrail System prevents semantic drift during continual learning by detecting conflicts using SID (Semantic Inconsistency Detector) and injecting symbolically grounded facts to stabilize the semantic space.

## Architecture

```
Training Batch → SID Conflict Detection → IF conflict THEN Guardrail Generation ELSE No Action
                                                ↓
                                    Augmented Batch (Original + Guardrails)
```

### Key Components

1. **GuardrailGenerator** (`guardrail/guardrail_generator.py`)
   - Generates 2-4 symbolically grounded facts when conflict detected
   - Three guardrail strategies:
     - **General rule reinforcement**: Parent class capabilities (e.g., "Birds can fly")
     - **Sibling examples**: Similar entities with same capability (e.g., "Eagles can fly", "Sparrows can fly")
     - **Hierarchy preservation**: Taxonomic relationships (e.g., "Penguins are birds")

2. **GuardrailController** (`guardrail/guardrail_controller.py`)
   - SID-gated controller implementing hard gating
   - **IF** conflict detected → activate guardrails
   - **ELSE** → no guardrails (normal training)
   - Augments training batches with generated facts

## Usage

### Basic Usage

```python
from guardrail import GuardrailController

# Initialize controller
controller = GuardrailController(max_guardrails=4)

# Knowledge base (existing facts)
knowledge_base = [
    "Birds can fly.",
    "Penguins are birds.",
    "Penguins cannot fly."
]

# Process training batch
batch = ["Penguins can fly."]  # Conflicting sentence
result = controller.process_batch(batch, knowledge_base)

# Check results
if result.has_conflict:
    print(f"Conflict detected!")
    print(f"Guardrails: {result.guardrail_samples}")
    # Output:
    # ['Birds can fly.', 'Eagles can fly.', 'Sparrows can fly.', 'Penguins are birds.']
    
    # Augmented batch for training
    augmented = result.original_samples + result.guardrail_samples
    # Train model on augmented batch
```

### Advanced Usage

```python
from sid import SemanticInconsistencyDetector
from guardrail import GuardrailGenerator, GuardrailController

# Custom SID with specific KB
sid = SemanticInconsistencyDetector(kb_path='custom_kb.json')

# Custom generator with different KB
generator = GuardrailGenerator(kb_path='guardrail_kb.json')

# Controller with custom components
controller = GuardrailController(
    sid=sid,
    generator=generator,
    max_guardrails=3,
    enable_guardrails=True
)
```

## Features

### 1. Hard SID-Gating

Guardrails **only** activate when conflict detected:

```python
# Clean batch (no conflict)
batch = ["Eagles have sharp talons."]
result = controller.process_batch(batch, kb)
assert result.has_conflict == False
assert len(result.guardrail_samples) == 0  # No guardrails

# Conflict batch
batch = ["Penguins can fly."]
result = controller.process_batch(batch, kb)
assert result.has_conflict == True
assert len(result.guardrail_samples) > 0  # Guardrails added
```

### 2. Symbolic Grounding

All guardrails are grounded in structured knowledge:

```python
generator = GuardrailGenerator('sid/knowledge_base.json')
facts = generator.generate(
    conflict_entity="penguin",
    conflict_relation="/r/CapableOf",
    conflict_object="fly",
    max_facts=4
)

# Output (4 facts):
# 1. "Birds can fly." (general rule from KB)
# 2. "Eagles can fly." (sibling from KB)
# 3. "Sparrows can fly." (sibling from KB)
# 4. "Penguins are birds." (hierarchy from KB)
```

### 3. Natural Language Output

All guardrails are natural language sentences:

```python
# NO symbolic notation like:
# ❌ "/c/en/bird /r/CapableOf /c/en/fly"

# YES natural language:
# ✓ "Birds can fly."
```

### 4. Budget Control

Guardrails respect 2-4 fact budget per conflict:

```python
controller = GuardrailController(max_guardrails=3)
result = controller.process_batch(["Penguins can fly."], kb)
assert len(result.guardrail_samples) <= 3
```

## Design Principles

### 1. Training-Time Augmentation

Guardrails are **data-level** interventions, NOT:
- ❌ Parameter freezing
- ❌ Loss regularization
- ❌ Gradient manipulation

They work by **augmenting training batches** with symbolic facts.

### 2. Hard Gating

Guardrails use **hard gating** via SID:
- `IF` conflict detected → activate guardrails
- `ELSE` → no guardrails (normal training)

This prevents over-intervention and maintains learning efficiency.

### 3. Positive Support

Guardrails provide **positive supporting facts**:
- ✓ "Birds can fly" (general rule)
- ✓ "Eagles can fly" (positive example)
- ✓ "Penguins are birds" (hierarchy)

NOT negative corrections:
- ❌ "Penguins cannot fly" (negation)

### 4. Semantic Stability

Guardrails stabilize semantic space against drift:
- **Exception overwriting**: New exception contradicts general rule
- **Over-generalization**: Model extends exception to entire category
- **Hierarchy collapse**: Taxonomic relationships degraded
- **Delayed drift**: Gradual semantic shift over tasks

## API Reference

### GuardrailGenerator

```python
class GuardrailGenerator:
    def __init__(self, kb_path: str = 'sid/knowledge_base.json'):
        """Initialize with knowledge base."""
        
    def generate(
        self,
        conflict_entity: str,
        conflict_relation: str,
        conflict_object: str,
        max_facts: int = 4
    ) -> List[GuardrailFact]:
        """
        Generate 2-4 guardrail facts for a conflict.
        
        Args:
            conflict_entity: Subject (e.g., "penguin")
            conflict_relation: Relation (e.g., "/r/CapableOf")
            conflict_object: Object (e.g., "fly")
            max_facts: Maximum facts (2-4 recommended)
            
        Returns:
            List of GuardrailFact objects
        """
```

### GuardrailController

```python
class GuardrailController:
    def __init__(
        self,
        sid: Optional[SemanticInconsistencyDetector] = None,
        generator: Optional[GuardrailGenerator] = None,
        max_guardrails: int = 4,
        enable_guardrails: bool = True
    ):
        """Initialize SID-gated controller."""
        
    def process_batch(
        self,
        batch: List[str],
        knowledge_base: List[str]
    ) -> TrainingBatch:
        """
        Process batch with SID-gated guardrail augmentation.
        
        Args:
            batch: Training sentences
            knowledge_base: Existing knowledge for conflict detection
            
        Returns:
            TrainingBatch with optional guardrails
        """
```

### Data Models

```python
@dataclass
class GuardrailFact:
    sentence: str           # Natural language sentence
    fact_type: str          # 'general_rule', 'sibling_example', 'hierarchy'
    source_relation: str    # Source relation type
    entities: List[str]     # Entities involved
    confidence: float       # Confidence score (0-1)

@dataclass
class TrainingBatch:
    original_samples: List[str]    # Original training sentences
    guardrail_samples: List[str]   # Generated guardrail sentences
    has_conflict: bool             # Whether conflict detected
    conflict_info: Dict[str, Any]  # Conflict details
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest test_guardrail.py -v

# Run specific test class
python -m pytest test_guardrail.py::TestGuardrailGenerator -v

# Run with coverage
python -m pytest test_guardrail.py --cov=guardrail --cov-report=html
```

Test coverage:
- **GuardrailGenerator**: 7 tests (entity normalization, fact generation, budget control)
- **GuardrailController**: 5 tests (conflict detection, hard gating, augmentation)
- **Integration**: 2 tests (end-to-end workflow, statistics)

## Examples

### Example 1: Exception Overwriting Protection

```python
# Scenario: Learning "Penguins cannot fly" after "Birds can fly"
kb = ["Birds can fly.", "Penguins are birds."]
batch = ["Penguins cannot fly."]

result = controller.process_batch(batch, kb)
# Guardrails:
# - "Birds can fly." (reinforces general rule)
# - "Eagles can fly." (positive sibling examples)
# - "Sparrows can fly."
# - "Penguins are birds." (preserves hierarchy)
```

### Example 2: Over-Generalization Prevention

```python
# Scenario: Model might generalize "cannot fly" to all birds
kb = ["Birds can fly.", "Penguins cannot fly."]
batch = ["Penguins are flightless."]

result = controller.process_batch(batch, kb)
# Guardrails provide counter-examples:
# - "Eagles can fly."
# - "Sparrows can fly."
```

### Example 3: No Conflict (No Guardrails)

```python
# Scenario: Consistent new knowledge
kb = ["Birds can fly.", "Penguins are birds."]
batch = ["Eagles have sharp talons."]

result = controller.process_batch(batch, kb)
assert result.has_conflict == False
assert len(result.guardrail_samples) == 0
# Normal training proceeds without intervention
```

## Performance

- **Conflict detection**: ~5-10ms per sentence (SID)
- **Guardrail generation**: ~20-30ms per conflict (KB lookup)
- **Total overhead**: <50ms per conflicting batch
- **Storage**: O(1) (no persistent state)
- **Memory**: O(k) where k = knowledge base size

## Citation

```bibtex
@article{naik2025sgcl,
  title={SGCL: Semantic-Guided Continual Learning with Symbolic Guardrails},
  author={Naik, Mithun},
  journal={Capstone Project},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Related

- **SID Module**: `sid/` - Semantic Inconsistency Detector
- **SeCA Dataset**: `sid/seca_publication.py` - Publication-ready evaluation dataset
- **Knowledge Base**: `sid/knowledge_base.json` - ConceptNet-based structured knowledge

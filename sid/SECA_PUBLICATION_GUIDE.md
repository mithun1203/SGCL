# SeCA Publication Dataset v2.0

## Overview

**SeCA (Semantic Consistency Aware) Dataset** is a publication-ready benchmark for evaluating continual learning systems on semantic conflict detection.

- **Total Samples**: 320
- **Tasks**: 8 (40 samples each)
- **Version**: 2.0 (Publication Ready)
- **Author**: Mithun Naik
- **Project**: SGCL Capstone

## Dataset Statistics

### Label Distribution
- **Non-conflict**: 240 samples (75.0%)
- **Conflict**: 60 samples (18.8%)
- **Ambiguous**: 20 samples (6.2%)

### Difficulty Distribution
- **Easy**: 140 samples (43.8%)
- **Medium**: 100 samples (31.2%)
- **Hard**: 80 samples (25.0%)

### Conflict Types
- **Direct Contradiction**: 20 samples
- **Paraphrase Conflict**: 20 samples
- **Multi-hop Reasoning**: 20 samples
- **Delayed Conflict**: 20 samples

## Task Structure

### T1: General Rules (Base Semantics)
**40 samples** | All non-conflict | Difficulty: Easy

Establishes baseline knowledge with universal facts:
- Animal properties (birds, fish, mammals)
- Vehicle characteristics
- General world knowledge

**Example**: "Birds can fly.", "Fish live in water.", "Mammals are warm-blooded."

---

### T2: Hierarchy / Taxonomy
**40 samples** | All non-conflict | Difficulty: Easy

is-a and part-of relations:
- Penguins are birds
- Sharks are fish
- Electric cars are vehicles
- Dogs are animals

**Example**: "Penguins are birds.", "Sharks are fish.", "Electric cars are vehicles."

---

### T3: Attribute Inheritance
**40 samples** | All non-conflict | Difficulty: Easy

Properties inherited from parent classes:
- Penguins have wings (from birds)
- Sharks have gills (from fish)
- Electric cars have wheels (from vehicles)

**Example**: "Penguins have wings.", "Sharks have gills.", "Electric cars have four wheels."

---

### T4: Exceptions (CORE TEST)
**40 samples** | All non-conflict | Difficulty: Medium

Exceptions to general rules - the core challenge:
- Flightless birds (penguins, ostriches, emus, kiwis)
- Electric vehicle exceptions (no gasoline, no exhaust)
- Fish that breathe air (lungfish)
- Egg-laying mammals (platypus, echidna)

**Example**: "Penguins cannot fly.", "Electric cars do not use gasoline.", "Bats are the only mammals that can truly fly."

**Purpose**: These are NOT conflicts - they establish valid exceptions that must be remembered.

---

### T5: Direct Contradictions
**40 samples** | 20 conflict + 20 non-conflict | Difficulty: Medium-Hard

Intentional conflicts with established knowledge:

**Conflict Examples**:
- "Penguins can fly." ← conflicts with T4: "Penguins cannot fly."
- "Electric cars use petrol." ← conflicts with T4: "Electric cars do not use gasoline."
- "Whales are fish." ← conflicts with T1: "Whales are aquatic mammals."

**Non-conflict Examples**:
- "Eagles have sharp talons."
- "Pandas eat bamboo."
- "Giraffes have long necks."

---

### T6: Paraphrase & QA Conflicts
**40 samples** | 20 conflict + 20 non-conflict | Difficulty: Easy-Hard

Same knowledge in different surface forms:

**Conflict Examples**:
- "Can penguins fly?" ← conflicts with T4
- "Are penguins capable of flight?" ← conflicts with T4
- "Do electric cars run on gasoline?" ← conflicts with T4

**Non-conflict Examples**:
- "Can birds fly?" 
- "Are birds capable of flight?"
- "Do fish live in water?"

**Challenge**: Detecting conflicts across different linguistic expressions (questions, statements, paraphrases).

---

### T7: Multi-hop Logical Reasoning
**40 samples** | 20 conflict + 20 non-conflict | Difficulty: Medium-Hard

Requires combining multiple facts:

**Conflict Examples**:
- "Penguins can fly because they are birds."
  - Reasoning: Birds can fly (T1) + Penguins are birds (T2) → BUT penguins cannot fly (T4)
  
- "Whales breathe through gills because they live in water."
  - Reasoning: Fish have gills (T1) + Whales live in water → BUT whales are mammals and breathe air

**Non-conflict Examples**:
- "Sparrows can fly because they are birds."
  - Reasoning: Birds can fly (T1) + Sparrows are birds (T2) → Valid!

**Each sample includes**:
- `reasoning_chain`: Step-by-step logical progression
- `conflicts_with`: List of conflicting facts from previous tasks

---

### T8: Delayed Contradictions (HARDEST)
**40 samples** | 20 ambiguous + 20 non-conflict | Difficulty: Hard-Medium

Conflicts that appear after several tasks - tests long-term memory:

**Ambiguous Examples** (testing catastrophic forgetting):
- "Penguins can soar through the sky."
  - References: T2 (penguins are birds), T1 (birds fly), T4 (penguins cannot fly)
  
- "Electric cars refuel at gas stations."
  - References: T2 (electric cars are vehicles), T1 (vehicles need energy), T4 (no gasoline)

**Non-conflict Examples**:
- "Eagles still have sharp vision many tasks later."
- "Dogs remain capable of barking across all tasks."

**Purpose**: Tests if model remembers facts from T1-T4 after learning T5-T7.

---

## Annotation Format

Each sample includes:

```json
{
  "task_id": 1,
  "sample_id": 0,
  "sentence": "Birds can fly.",
  "label": "no_conflict",
  "conflicts_with": [],
  "conflict_type": "none",
  "entities": ["birds"],
  "relations": ["CapableOf"],
  "reasoning_chain": [],
  "difficulty": "easy"
}
```

### Fields:
- **task_id**: Task number (1-8)
- **sample_id**: Sample index within task
- **sentence**: The statement
- **label**: `no_conflict`, `conflict`, or `ambiguous`
- **conflicts_with**: List of conflicting statements from previous tasks
- **conflict_type**: Category of conflict
- **entities**: Named entities in the sentence
- **relations**: Semantic relations (CapableOf, IsA, HasA, etc.)
- **reasoning_chain**: Step-by-step reasoning for multi-hop samples
- **difficulty**: `easy`, `medium`, or `hard`

---

## Evaluation Methodology

### Split Files
Located in `evaluation_splits/`:
- `non_conflict_split.json` (240 samples)
- `conflict_split.json` (60 samples)
- `ambiguous_split.json` (20 samples)
- `all_split.json` (320 samples)

### Metrics

1. **Conflict Detection Accuracy**
   - True Positive Rate (detecting conflicts)
   - True Negative Rate (accepting valid statements)
   - F1-Score on conflict class

2. **Catastrophic Forgetting**
   - Accuracy on T1-T4 facts after learning T5-T8
   - Measured using delayed contradictions (T8)

3. **Multi-hop Reasoning**
   - Accuracy on T7 samples requiring fact combination
   - Reasoning chain correctness

4. **Paraphrase Robustness**
   - Accuracy on T6 across surface form variations
   - Consistency across rephrasings

---

## Knowledge Sources

All knowledge is **curated from authoritative sources** - no generated or invented facts:

- **ConceptNet**: Common-sense relations (CapableOf, HasProperty, IsA)
- **WordNet**: Taxonomic hierarchies and definitions
- **DBpedia**: Factual knowledge about entities
- **Wikipedia**: Exception cases and special properties

**No LLM-generated content** - ensures dataset quality and reproducibility.

---

## Usage

### Load Dataset

```python
from sid.seca_publication import SeCAPublicationDataset

# Load from file
dataset = SeCAPublicationDataset.load("sid/seca_publication_dataset.json")

# Access tasks
for task in dataset.tasks:
    print(f"Task {task.task_id}: {task.name}")
    print(f"  Samples: {len(task.samples)}")

# Access samples
for sample in dataset.tasks[0].samples:
    print(f"  {sample.sentence} -> {sample.label}")
```

### Validation

```bash
python -m sid.validate_publication
```

Output:
- ✓ 10/10 validation checks passed
- Detailed statistics
- Evaluation splits created

---

## Experimental Protocol

### Continual Learning Setup

**Task Sequence**: T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8

**Training**:
1. Train on Task 1 (40 samples)
2. Evaluate on Task 1
3. Train on Task 2 (40 samples)
4. Evaluate on Tasks 1-2 (test forgetting)
5. Continue sequentially...

**Final Evaluation**:
- Test on all 320 samples
- Measure backward transfer (T1-T4 after T5-T8)
- Measure forward transfer (conflict detection improves?)

### Baseline Models

Recommended baselines:
1. **Fine-tuning**: Sequential fine-tuning (expect catastrophic forgetting)
2. **EWC**: Elastic Weight Consolidation
3. **Replay**: Experience Replay with memory buffer
4. **GEM**: Gradient Episodic Memory
5. **A-GEM**: Averaged GEM
6. **PackNet**: Parameter masking
7. **ProgressiveNN**: Progressive Neural Networks

### Expected Results

**Good Continual Learner**:
- High accuracy on T1-T4 even after T5-T8
- Detects conflicts in T5-T6 using T1-T4 memory
- Handles multi-hop reasoning in T7
- Low catastrophic forgetting on T8

**Poor Continual Learner**:
- Forgets T1-T4 after learning T5-T8
- Cannot detect conflicts (treats contradictions as new facts)
- Fails multi-hop reasoning
- High catastrophic forgetting

---

## Key Challenges

### 1. Exception Handling (T4)
Model must learn that **exceptions are not conflicts**:
- "Birds can fly" (T1) is true
- "Penguins cannot fly" (T4) is ALSO true (exception, not conflict)
- "Penguins can fly" (T5) is FALSE (conflict!)

### 2. Long-term Memory (T8)
After learning 7 tasks (280 samples), can model still remember facts from T1?

### 3. Multi-hop Inference (T7)
Must combine facts across tasks:
- Know: "Birds fly" (T1) + "Penguins are birds" (T2)
- Infer: Penguins should fly
- Recall: "Penguins cannot fly" (T4)
- Detect: Conflict!

### 4. Surface Form Variation (T6)
Same conflict expressed differently:
- "Penguins can fly"
- "Can penguins fly?"
- "Are penguins capable of flight?"
- "Penguins possess the capability to fly"

All conflict with "Penguins cannot fly" (T4).

---

## Citation

```bibtex
@dataset{naik2024seca,
  title={SeCA: Semantic Consistency Aware Dataset for Continual Learning},
  author={Naik, Mithun},
  year={2024},
  version={2.0},
  url={https://github.com/mithunnaik/sgcl},
  note={SGCL Capstone Project}
}
```

---

## Files Structure

```
sid/
├── seca_publication.py              # Dataset creation script
├── seca_publication_dataset.json    # Full 320-sample dataset
├── validate_publication.py          # Validation script
├── evaluation_splits/
│   ├── non_conflict_split.json      # 240 non-conflict samples
│   ├── conflict_split.json          # 60 conflict samples
│   ├── ambiguous_split.json         # 20 ambiguous samples
│   └── all_split.json               # All 320 samples
└── SECA_PUBLICATION_GUIDE.md        # This file
```

---

## Quality Assurance

### ✓ Validation Passed (10/10 checks)

1. ✓ Total 320 samples
2. ✓ 8 tasks present
3. ✓ 40 samples per task
4. ✓ Non-conflict ≥ 100 (240)
5. ✓ Conflict ≥ 40 (60)
6. ✓ Ambiguous ≥ 20 (20)
7. ✓ ≥4 conflict types (4)
8. ✓ All tasks have 40 samples
9. ✓ All 8 task types present
10. ✓ Conflicts annotated (80/80)

### Dataset is PUBLICATION-READY ✓

---

## Contact

**Author**: Mithun Naik  
**Project**: SGCL (Semantic-Guided Continual Learning) Capstone  
**Institution**: [Your University]  
**Email**: [Your Email]

---

## License

This dataset is released for academic and research purposes.

---

**Last Updated**: December 22, 2024  
**Version**: 2.0 (Publication Ready)

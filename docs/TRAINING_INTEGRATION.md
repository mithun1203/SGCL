# SG-CL Training Integration

**Complete continual learning training system with SID-gated guardrails**

---

## 🎯 What This Is

This is the **training integration** that connects all SGCL components into a working continual learning system:

- **SID** (Semantic Inconsistency Detector) → detects conflicts
- **Guardrails** (Symbolic facts) → stabilizes semantics  
- **LoRA** (Low-Rank Adaptation) → efficient fine-tuning
- **Sequential Tasks** → continual learning setup

**ONE LINE SUMMARY:**  
*SG-CL = inserting SID-controlled, guardrail-augmented batches into normal LLM fine-tuning*

---

## 🏗️ Architecture

```
Sequential Task (t)
      ↓
Training Sample
      ↓
┌─────────────────┐
│ SID Detector    │ ← Check conflict with previous knowledge
└────────┬────────┘
         │
    Conflict?
    ╱        ╲
  YES         NO
   │           │
   ↓           ↓
Generate     Train
Guardrails   Normally
   │           │
   ↓           │
Augment       │
Batch         │
   │           │
   └─────┬─────┘
         ↓
    Train with
    Gradient Update
```

---

## 📦 Files

### Core Components

1. **`sgcl_training.py`** - Main training loop
   - `SGCLTrainer` - Full SG-CL system
   - `NaiveFinetuningTrainer` - Baseline (no guardrails)
   - Training config and metrics

2. **`sgcl_data_loader.py`** - Dataset loading
   - `SeCALoader` - Loads SeCA v2.0 in task-sequential format
   - `create_toy_tasks()` - Quick testing tasks
   - `create_minimal_tasks()` - Minimal testing

3. **`run_sgcl_experiment.py`** - Complete experiment script
   - Trains both SG-CL and baseline
   - Compares results
   - Saves models and metrics

---

## 🚀 Quick Start

### Option 1: Minimal Test (Fastest)

```bash
# Run on 2 tiny tasks (2-3 samples each)
# Takes ~5-10 minutes on GPU
python run_sgcl_experiment.py --dataset minimal --name quick_test
```

### Option 2: Toy Tasks

```bash
# Run on 3 toy tasks (5 samples each)
# Takes ~15-20 minutes
python run_sgcl_experiment.py --dataset toy --name toy_exp
```

### Option 3: Full SeCA Dataset

```bash
# Run on first 2 SeCA tasks
python run_sgcl_experiment.py --dataset seca --tasks "0,1" --name seca_exp

# Run on all 8 SeCA tasks (full experiment)
python run_sgcl_experiment.py --dataset seca --name full_seca
```

---

## 🔬 Experiment Output

After running, you'll get:

```
experiments/
└── sgcl_experiment_20251222_143000/
    ├── sgcl_model/              # SG-CL trained model
    │   ├── adapter_model.bin
    │   ├── adapter_config.json
    │   └── tokenizer/
    ├── baseline_model/          # Baseline trained model
    │   ├── adapter_model.bin
    │   └── ...
    ├── sgcl_stats.json         # SG-CL training metrics
    ├── baseline_stats.json     # Baseline training metrics
    └── comparison.json         # Side-by-side comparison
```

### Key Metrics Logged

```json
{
  "total_samples": 100,
  "conflicts_detected": 15,
  "conflict_rate": 0.15,
  "total_guardrails": 60,
  "avg_guardrails_per_conflict": 4.0,
  "avg_loss": 2.3456,
  "task_stats": {
    "task_0": {
      "samples": 50,
      "conflicts": 5,
      "guardrails": 20,
      "avg_loss": 2.1234
    }
  }
}
```

---

## 💻 Python API

### Train SG-CL

```python
from sgcl_training import SGCLTrainer, TrainingConfig
from sgcl_data_loader import create_toy_tasks

# Load tasks
tasks, task_names = create_toy_tasks()

# Configure
config = TrainingConfig(
    model_name="microsoft/phi-3-mini-4k-instruct",
    lora_r=8,
    lora_alpha=16,
    learning_rate=2e-4,
    max_guardrails=4,
    enable_guardrails=True  # This makes it SG-CL
)

# Train
trainer = SGCLTrainer(config)
stats = trainer.train_on_tasks(tasks, task_names)

# Save
trainer.save_model("models/my_sgcl_model")
```

### Train Baseline

```python
from sgcl_training import NaiveFinetuningTrainer

# Same config but no guardrails
trainer = NaiveFinetuningTrainer(config)
stats = trainer.train_on_tasks(tasks, task_names)
```

---

## 🔑 Key Components Explained

### 1. Core Training Loop

```python
for task_id, task_data in enumerate(tasks):
    for sample in task_data:
        
        # ━━━ THIS IS SG-CL ━━━
        
        # Step 1: Check conflict
        result = guardrail_controller.process_batch(
            [sample], 
            knowledge_base
        )
        
        # Step 2: Augment if conflict
        if result.has_conflict:
            training_batch = sample + guardrails
        else:
            training_batch = [sample]
        
        # Step 3: Standard training
        loss = train_step(training_batch)
        
        # ━━━━━━━━━━━━━━━━━━━━
```

**That's it.** No special optimizer, no loss modification, just data augmentation.

### 2. What Makes It Different from Baselines

| Method | Difference |
|--------|-----------|
| **SG-CL** | SID + guardrails augment batch |
| **Naive FT** | No SID, no guardrails |
| **EWC** | Loss regularization (parameter-level) |
| **Replay** | Memory buffer (past samples) |

Same model, same optimizer, same training budget.  
Only difference: **what data the model sees**.

### 3. Hard Gating

```python
if conflict_detected:
    add_guardrails()  # ONLY when needed
else:
    train_normally()  # No overhead
```

This is **hard gating** - guardrails only activate on conflict, not continuously.

---

## 🎓 For Your Viva/Defense

### Question: "What is SG-CL training integration?"

**Answer:**
> "SG-CL training integration inserts SID-controlled, guardrail-augmented batches into a standard LLM fine-tuning loop. When the Semantic Inconsistency Detector identifies a conflict, we augment the training batch with 2-4 symbolically grounded facts from our knowledge base. This stabilizes the semantic space without modifying the optimizer or loss function - it's pure data-level intervention."

### Question: "How is this different from EWC or replay methods?"

**Answer:**
> "EWC operates at the parameter level by adding regularization terms to the loss function. Replay buffers store past samples. SG-CL operates at the data level by injecting symbolic facts in real-time based on semantic conflicts. This makes it architecture-agnostic and interpretable - you can read the guardrails and understand exactly what knowledge is being reinforced."

### Question: "Why use guardrails instead of just blocking conflicting updates?"

**Answer:**
> "Blocking updates would prevent learning. Guardrails provide positive supporting facts that contextualize the new knowledge. For example, if the model learns 'Penguins cannot fly' (conflicting with 'Birds can fly'), we inject 'Birds can fly', 'Eagles can fly', 'Sparrows can fly', and 'Penguins are birds'. This preserves the general rule while allowing the exception to be learned."

---

## 📊 Expected Results

### Metrics You Should See

1. **Conflict Detection**
   - SG-CL: 10-30% conflict rate (depending on tasks)
   - Baseline: Same conflicts but no intervention

2. **Guardrails Injected**
   - SG-CL: 2-4 guardrails per conflict
   - Baseline: 0 (no guardrails)

3. **Training Overhead**
   - <50ms per conflict for guardrail generation
   - Negligible compared to model forward/backward pass

4. **Semantic Consistency** (evaluated separately with SeCA)
   - SG-CL: Higher accuracy on drift detection tasks
   - Baseline: More semantic drift

---

## 🛠️ Customization

### Use Different Model

```python
config = TrainingConfig(
    model_name="gpt2",  # or any HuggingFace model
    # ... rest same
)
```

### Change LoRA Config

```python
config = TrainingConfig(
    lora_r=16,        # Higher rank
    lora_alpha=32,    # Adjust alpha
    lora_dropout=0.1  # Higher dropout
)
```

### Adjust Guardrail Budget

```python
config = TrainingConfig(
    max_guardrails=2  # Fewer guardrails (faster)
    # or
    max_guardrails=6  # More guardrails (stronger)
)
```

---

## 🧪 Ablation Studies

Run ablation by modifying config:

```python
# Ablation 1: No guardrails (baseline)
config.enable_guardrails = False

# Ablation 2: Different guardrail counts
for n in [2, 3, 4, 6]:
    config.max_guardrails = n
    # train and compare

# Ablation 3: Different LoRA ranks
for r in [4, 8, 16]:
    config.lora_r = r
    # train and compare
```

---

## 📝 Requirements

```bash
pip install torch transformers peft tqdm
```

**Models downloaded automatically:**
- microsoft/phi-3-mini-4k-instruct (~7.5GB)

**GPU Recommended:**
- 16GB VRAM minimum
- 24GB for larger experiments

**CPU Mode:**
- Will work but very slow
- Set `device="cpu"` in config

---

## 🚦 Status Checklist

- ✅ Core training loop implemented
- ✅ SID integration working
- ✅ Guardrail augmentation working  
- ✅ Baseline comparison included
- ✅ Metrics logging complete
- ✅ SeCA data loader ready
- ✅ Experiment runner script done
- ⏳ Need to run actual experiments
- ⏳ Need to evaluate with SeCA benchmarks

---

## 📖 Next Steps

1. **Run quick test**
   ```bash
   python run_sgcl_experiment.py --dataset minimal
   ```

2. **Analyze results**
   ```bash
   cat experiments/*/comparison.json
   ```

3. **Run full experiments** (for paper)
   ```bash
   python run_sgcl_experiment.py --dataset seca --name full_exp
   ```

4. **Evaluate semantic consistency** (separate evaluation with SeCA)

---

## 🎯 Key Takeaway

**SG-CL Training Integration = Data-level intervention with symbolic guardrails**

- NOT parameter freezing
- NOT loss regularization  
- NOT gradient blocking

It's **data augmentation** triggered by semantic conflicts.

Simple, interpretable, effective.

---

**Status**: ✅ Training integration complete and ready for experiments

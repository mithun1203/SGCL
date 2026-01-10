# SG-CL Training Engine - Kaggle Deployment Package

## 📦 Package Contents

This training package contains everything needed to run SG-CL experiments on Kaggle GPU:

### Core Training Files:
- `sgcl_trainer.py` - Main SG-CL training engine
- `baseline_trainers.py` - Baseline implementations (Naive, EWC, Replay)
- `requirements_kaggle.txt` - Required packages for Kaggle
- `kaggle_train_sgcl.ipynb` - Kaggle notebook for SG-CL training
- `kaggle_train_baselines.ipynb` - Kaggle notebook for baseline training

### Required from Main Project:
- `sid/` directory (entire SID module)
- `guardrail/` directory (entire guardrail module)
- `sid/seca_10k_final.json` (SeCA dataset)

## 🚀 Quick Start on Kaggle

### Step 1: Upload Files

Upload to Kaggle Notebook:
1. `sgcl_trainer.py`
2. `baseline_trainers.py`
3. Entire `sid/` directory
4. Entire `guardrail/` directory

### Step 2: Install Dependencies

```python
!pip install transformers peft accelerate bitsandbytes sentencepiece
```

### Step 3: Run Training

```python
# For SG-CL
from sgcl_trainer import SGCLTrainer, TrainingConfig

config = TrainingConfig(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    batch_size=4,
    num_epochs=3,
    enable_sid=True,
    enable_guardrails=True
)

trainer = SGCLTrainer(config)
trainer.train_sequential_tasks("sid/seca_10k_final.json", num_tasks=5)
```

## 📊 Training Configuration

### Recommended Settings for Kaggle GPU:

```python
TrainingConfig(
    # Model
    model_name="microsoft/Phi-3-mini-4k-instruct",  # ~3.8B params
    max_length=512,
    
    # LoRA (efficient fine-tuning)
    lora_r=16,                    # Rank
    lora_alpha=32,                # Scaling
    lora_dropout=0.1,
    
    # Training
    batch_size=4,                 # Adjust based on GPU memory
    learning_rate=2e-4,
    num_epochs=3,
    gradient_accumulation_steps=4,
    
    # SG-CL specific
    enable_sid=True,              # Enable conflict detection
    enable_guardrails=True,       # Enable guard-rail generation
    max_guardrails_per_conflict=4
)
```

### Memory Optimization Tips:

If you run out of memory:
1. Reduce `batch_size` to 2 or 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_length` to 256
4. Use `torch.float16` (already enabled)

## 🧪 Experimental Pipeline

### Phase 1: Train SG-CL
```python
trainer = SGCLTrainer(config)
trainer.train_sequential_tasks("sid/seca_10k_final.json", num_tasks=5)
```

### Phase 2: Train Baselines
```python
from baseline_trainers import NaiveFinetuning, EWCTrainer, ReplayBufferTrainer

# Naive
naive = NaiveFinetuning(config)
naive.train_sequential_tasks(dataset_path, num_tasks=5)

# EWC
ewc = EWCTrainer(config, ewc_lambda=5000)
ewc.train_sequential_tasks(dataset_path, num_tasks=5)

# Replay
replay = ReplayBufferTrainer(config, buffer_size=500)
replay.train_sequential_tasks(dataset_path, num_tasks=5)
```

### Phase 3: Evaluation (Next Step)
After training completes, implement evaluation module to compute:
- Semantic Consistency Score
- Contradiction Rate
- Forgetting Curves
- Comparison Plots

## 📁 Output Structure

Training produces:
```
sgcl_checkpoints/
├── task_0/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── metrics.json
├── task_1/
│   └── ...
├── task_2/
│   └── ...
└── training_summary.json
```

## 🔍 Monitoring Training

Key metrics logged:
- `loss`: Training loss per batch
- `conflicts_detected`: Number of conflicts found by SID
- `guardrails_generated`: Number of guard-rail facts injected
- `samples_processed`: Total samples trained on

## ⚙️ Algorithm Flow (SG-CL)

```
For each task:
  For each batch:
    1. Load batch from SeCA
       ↓
    2. SID checks each sample for conflicts
       ↓
    3. If conflict detected → Generate guard-rails
       ↓
    4. Augment batch with guard-rail facts
       ↓
    5. Tokenize augmented batch
       ↓
    6. Forward pass (compute loss)
       ↓
    7. Backward pass (gradient computation)
       ↓
    8. Update LoRA weights
       ↓
  Save checkpoint after task
```

## 📊 Expected Results

After training 5 tasks, you should have:
- 5 task-specific checkpoints
- Training metrics for each task
- Evidence of SID detecting conflicts
- Evidence of guard-rail generation

Example metrics:
```json
{
  "task_0": {
    "samples": 2000,
    "conflicts": 150,
    "guardrails": 450,
    "avg_loss": 2.34
  }
}
```

## 🐛 Troubleshooting

**OutOfMemoryError:**
- Reduce batch_size to 1
- Increase gradient_accumulation_steps to 8

**ImportError for sid/guardrail:**
- Ensure entire `sid/` and `guardrail/` directories are uploaded
- Check `sys.path.insert(0, '.')` is in notebook

**Slow training:**
- Enable GPU in Kaggle notebook settings
- Reduce num_epochs to 1 for testing

## 📝 Notes

- Training 5 tasks × 3 epochs × ~2000 samples/task = ~30K samples
- Estimated time: 2-4 hours on Kaggle GPU (T4)
- LoRA keeps memory usage low (~6GB for Phi-3-mini)
- SID runs on CPU during batch processing (minimal overhead)

## 🎯 Next Steps After Training

1. Download checkpoints from Kaggle
2. Implement evaluation module
3. Run SCP scoring
4. Generate comparison plots
5. Write results in paper

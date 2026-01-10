# 🚀 Kaggle Training - Quick Start

## 📦 Setup (2 Minutes)

### Step 1: Clone Repository in Kaggle

Open a Kaggle notebook and run:

```bash
# Clone your repository
!git clone https://github.com/mithun1203/SGCL.git
%cd SGCL
```

That's it! All files are ready.

### Step 2: Install Dependencies

```bash
!pip install -q transformers peft accelerate bitsandbytes sentencepiece
```

## 🎯 Training Scripts

### Option 1: SG-CL Training (Our Method)

```bash
# In Kaggle notebook cell
!python training/kaggle_train_sgcl.py
```

**What it does:**
- Trains Phi-3-mini with SG-CL algorithm
- Uses SID for conflict detection
- Generates guard-rails during training
- Saves checkpoints to `sgcl_checkpoints/`

**Expected output:**
```
📦 Installing required packages...
🖥️  Device: Tesla T4
GPU Available: True
GPU Memory: 15.11 GB

⚙️  Configuration:
  Model: microsoft/Phi-3-mini-4k-instruct
  Batch size: 4
  Epochs per task: 3
  SID enabled: True
  Guard-rails enabled: True

🎓 STARTING SG-CL TRAINING
======================================================================
Training Task 0...
  Step 10 | Loss: 2.4521 | Conflicts: 23 | Guard-rails: 87
  Step 20 | Loss: 2.1834 | Conflicts: 19 | Guard-rails: 72
  ...
✅ Task 0 complete! (avg loss: 1.8234)

Training Task 1...
  ...

✅ SG-CL TRAINING COMPLETE!
```

### Option 2: Baseline Training (For Comparison)

```bash
# In Kaggle notebook cell
!python training/kaggle_train_baselines.py
```

**What it does:**
- Trains 3 baseline methods sequentially:
  1. Naive Fine-tuning (no CL)
  2. EWC (regularization-based)
  3. Replay Buffer (memory-based)
- Each uses same configuration as SG-CL
- Saves separate checkpoints for each

**Expected output:**
```
🔵 BASELINE 1: NAIVE FINE-TUNING
======================================================================
Training Task 0...
  Step 10 | Loss: 2.4823
  ...
✅ Naive fine-tuning complete!

🟢 BASELINE 2: EWC
======================================================================
Computing Fisher Information Matrix...
Training Task 0...
  ...
✅ EWC training complete!

🟡 BASELINE 3: REPLAY BUFFER
======================================================================
Training Task 0...
  Storing 100 samples in replay buffer
  ...
✅ Replay buffer training complete!

📊 BASELINE TRAINING SUMMARY
Final task average losses:
  Naive       : 2.8934
  EWC         : 2.3421
  Replay      : 2.1876
```

## ⚙️ Kaggle Notebook Setup

### Complete Notebook Template

```python
# Cell 1: Setup
!git clone https://github.com/mithun1203/SGCL.git
%cd SGCL
!pip install -q transformers peft accelerate bitsandbytes sentencepiece

# Cell 2: Run SG-CL Training
!python training/kaggle_train_sgcl.py

# Cell 3 (Optional): Run Baseline Comparison
# !python training/kaggle_train_baselines.py

# Cell 4: Zip and Download Results
!zip -r sgcl_results.zip sgcl_checkpoints/
# Download from Kaggle output panel
```

That's the entire setup! No file uploads needed.

## 📊 Expected Training Time

**On Kaggle T4 GPU (15GB VRAM):**

- **SG-CL**: ~2.5 hours (5 tasks × 3 epochs with conflict detection)
- **Baselines**: ~6 hours total (3 methods × 2 hours each)

**Memory usage:**
- Phi-3-mini (3.8B params) + LoRA: ~6GB
- Batch size 4 + gradient accumulation: ~8GB
- Peak memory: ~10GB (safe for T4)

## 🔧 If You Hit Memory Errors

Edit configuration in the training scripts:

```python
# Reduce batch size
config = TrainingConfig(
    batch_size=2,  # was 4
    gradient_accumulation_steps=8,  # was 4
    # ... rest stays same
)
```

## 📈 Monitoring Training

Kaggle will show real-time output:
- Loss curves
- Conflicts detected (SG-CL only)
- Guard-rails generated (SG-CL only)
- GPU memory usage
- Step timing

## 💾 Output Files

After training completes:

### SG-CL:
```
sgcl_checkpoints/
├── task_0/
│   └── adapter_model.bin  (LoRA weights)
├── task_1/
├── task_2/
├── task_3/
├── task_4/
├── final_model/
└── training_summary.json  (metrics)
```

### Baselines:
```
naive_checkpoints/    (same structure)
ewc_checkpoints/      (same structure)
replay_checkpoints/   (same structure)
```

## 🎯 Success Criteria

Training is successful if:
- ✅ All 5 tasks complete without errors
- ✅ Loss decreases over epochs (< 2.0 final loss)
- ✅ SG-CL detects conflicts (>0 per batch)
- ✅ Guard-rails generated (>0 per conflict)
- ✅ Checkpoints saved for all tasks

## 🐛 Troubleshooting

**Error: CUDA out of memory**
→ Reduce `batch_size` to 2 or 1

**Error: Cannot load dataset**
→ Check `seca_10k_final.json` is in `sid/` directory (should be auto-cloned)

**Error: Module not found (SID/Guardrail)**
→ Make sure you ran `%cd SGCL` to enter the repository directory

**Slow training (>5 hours)**
→ Normal for first run; Kaggle caches model after download

## 📞 Need Help?

Check these files:
- `training/README.md` - Full documentation
- `training/sgcl_trainer.py` - Implementation details
- `training/baseline_trainers.py` - Baseline implementations

---

**Status**: Ready to deploy on Kaggle GPU 🚀

"""
📊 Baseline Comparison Training on Kaggle GPU
==============================================

Trains three baseline methods for comparison with SG-CL:
1. Naive Fine-tuning (no CL mechanism)
2. EWC (Elastic Weight Consolidation)
3. Replay Buffer

Upload required files:
- training/baseline_trainers.py
- sid/seca_10k_final.json
"""

# ============================================================================
# SETUP
# ============================================================================

import sys
import torch
import json
from pathlib import Path

print("📦 Checking GPU availability...")

# Check GPU
print(f"\n🖥️  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# IMPORT BASELINE TRAINERS
# ============================================================================

print("\n📥 Loading baseline trainers...")
from baseline_trainers import NaiveFinetuning, EWCTrainer, ReplayBufferTrainer, TrainingConfig

print("✓ Baseline trainers loaded")

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

config = TrainingConfig(
    # Model settings
    model_name="microsoft/Phi-3-mini-4k-instruct",
    max_length=512,
    
    # LoRA settings
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    
    # Training settings
    batch_size=4,
    learning_rate=2e-4,
    num_epochs=3,
    warmup_steps=100,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    
    # Disable SID/Guardrails for baselines
    enable_sid=False,
    enable_guardrails=False,
    
    log_every_n_steps=10,
    save_every_n_steps=500
)

print("\n⚙️  Configuration:")
print(f"  Model: {config.model_name}")
print(f"  Batch size: {config.batch_size}")
print(f"  Epochs per task: {config.num_epochs}")

# ============================================================================
# BASELINE 1: NAIVE FINE-TUNING
# ============================================================================

print("\n" + "="*70)
print("🔵 BASELINE 1: NAIVE FINE-TUNING")
print("="*70)

config.output_dir = "./naive_checkpoints"
naive_trainer = NaiveFinetuning(config)

naive_trainer.train_sequential_tasks(
    dataset_path="sid/seca_10k_final.json",
    num_tasks=5
)

print("\n✅ Naive fine-tuning complete!")
print("📁 Checkpoints: naive_checkpoints/")

# ============================================================================
# BASELINE 2: EWC (ELASTIC WEIGHT CONSOLIDATION)
# ============================================================================

print("\n" + "="*70)
print("🟢 BASELINE 2: EWC (Elastic Weight Consolidation)")
print("="*70)

config.output_dir = "./ewc_checkpoints"
ewc_trainer = EWCTrainer(
    config,
    ewc_lambda=1000.0  # Regularization strength
)

ewc_trainer.train_sequential_tasks(
    dataset_path="sid/seca_10k_final.json",
    num_tasks=5
)

print("\n✅ EWC training complete!")
print("📁 Checkpoints: ewc_checkpoints/")

# ============================================================================
# BASELINE 3: REPLAY BUFFER
# ============================================================================

print("\n" + "="*70)
print("🟡 BASELINE 3: REPLAY BUFFER")
print("="*70)

config.output_dir = "./replay_checkpoints"
replay_trainer = ReplayBufferTrainer(
    config,
    buffer_size=500  # Store 500 samples
)

replay_trainer.train_sequential_tasks(
    dataset_path="sid/seca_10k_final.json",
    num_tasks=5
)

print("\n✅ Replay buffer training complete!")
print("📁 Checkpoints: replay_checkpoints/")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 BASELINE TRAINING SUMMARY")
print("="*70)

summaries = []
for name, path in [
    ("Naive", "naive_checkpoints/training_summary.json"),
    ("EWC", "ewc_checkpoints/training_summary.json"),
    ("Replay", "replay_checkpoints/training_summary.json")
]:
    with open(path, 'r') as f:
        summary = json.load(f)
    summaries.append((name, summary))

# Compare final task losses
print("\nFinal task average losses:")
for name, summary in summaries:
    final_task = summary['task_metrics'][-1]
    avg_loss = sum(final_task['loss']) / len(final_task['loss'])
    print(f"  {name:12s}: {avg_loss:.4f}")

# ============================================================================
# DOWNLOAD INSTRUCTIONS
# ============================================================================

print("\n💾 All checkpoints saved:")
print("  - naive_checkpoints/")
print("  - ewc_checkpoints/")
print("  - replay_checkpoints/")

print("\nTo download:")
print("  1. Click Files icon (left sidebar)")
print("  2. Navigate to checkpoint directories")
print("  3. Right-click → Download")

print("\n🎉 All baseline training complete!")
print("\n📈 Next: Compare with SG-CL results using evaluation module")

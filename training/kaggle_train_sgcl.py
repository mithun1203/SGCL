"""
🚀 SG-CL Training on Kaggle GPU
===============================

This notebook runs the complete SG-CL training experiment.

Upload required files:
- training/sgcl_trainer.py
- training/baseline_trainers.py  
- sid/ (entire directory)
- guardrail/ (entire directory)
- sid/seca_10k_final.json
"""

# ============================================================================
# SETUP
# ============================================================================

print("📦 Installing required packages...")
!pip install -q transformers peft accelerate bitsandbytes sentencepiece

import sys
import torch
import json
from pathlib import Path

# Check GPU
print(f"\n🖥️  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# IMPORT SG-CL TRAINER
# ============================================================================

print("\n📥 Loading SG-CL training engine...")
from sgcl_trainer import SGCLTrainer, TrainingConfig

print("✓ SG-CL Trainer loaded successfully")

# ============================================================================
# CONFIGURATION
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
    batch_size=4,                    # Adjust if OOM
    learning_rate=2e-4,
    num_epochs=3,                    # 3 epochs per task
    warmup_steps=100,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    
    # SG-CL specific
    enable_sid=True,                 # ✅ Enable conflict detection
    enable_guardrails=True,          # ✅ Enable guard-rail generation
    max_guardrails_per_conflict=4,
    
    # Logging & checkpointing
    log_every_n_steps=10,
    save_every_n_steps=500,
    output_dir="./sgcl_checkpoints"
)

print("\n⚙️  Configuration:")
print(f"  Model: {config.model_name}")
print(f"  Batch size: {config.batch_size}")
print(f"  Epochs per task: {config.num_epochs}")
print(f"  SID enabled: {config.enable_sid}")
print(f"  Guard-rails enabled: {config.enable_guardrails}")

# ============================================================================
# INITIALIZE TRAINER
# ============================================================================

print("\n🚀 Initializing SG-CL Trainer...")
trainer = SGCLTrainer(config)
print("✓ Trainer initialized")

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*70)
print("🎓 STARTING SG-CL TRAINING")
print("="*70)

# Train on first 5 tasks from SeCA dataset
trainer.train_sequential_tasks(
    dataset_path="sid/seca_10k_final.json",
    num_tasks=5
)

print("\n" + "="*70)
print("✅ SG-CL TRAINING COMPLETE!")
print("="*70)

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n📊 Loading training summary...")
with open("sgcl_checkpoints/training_summary.json", 'r') as f:
    summary = json.load(f)

print(f"\nTotal tasks trained: {summary['num_tasks']}")
print(f"Total training steps: {summary['total_steps']}")

print("\nPer-task metrics:")
for i, task_metrics in enumerate(summary['task_metrics']):
    print(f"\n  Task {i}:")
    print(f"    Samples: {task_metrics['samples']}")
    print(f"    Conflicts detected: {task_metrics['conflicts']}")
    print(f"    Guard-rails generated: {task_metrics['guardrails']}")
    print(f"    Avg loss: {sum(task_metrics['loss'])/len(task_metrics['loss']):.4f}")

# ============================================================================
# DOWNLOAD CHECKPOINTS
# ============================================================================

print("\n💾 Checkpoints saved to: sgcl_checkpoints/")
print("\nTo download:")
print("  1. Click Files icon (left sidebar)")
print("  2. Navigate to sgcl_checkpoints/")
print("  3. Right-click → Download")

print("\n🎉 Training session complete!")

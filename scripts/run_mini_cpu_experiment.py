"""
CPU-Optimized Mini Experiment

Uses a tiny model (GPT-2) instead of Phi-3 for much faster testing on CPU.
Perfect for verifying the system works before running full experiments.
"""

import torch
from sgcl_training import SGCLTrainer, NaiveFinetuningTrainer, TrainingConfig
from sgcl_data_loader import create_minimal_tasks
import json
from pathlib import Path
from datetime import datetime

def run_mini_experiment():
    """
    Run a CPU-friendly mini experiment with GPT-2.
    
    This is MUCH faster than Phi-3 and perfect for testing.
    """
    print("="*70)
    print("  CPU-Optimized Mini Experiment")
    print("="*70)
    print("Using GPT-2 (124M params) instead of Phi-3 (3.8B params)")
    print("This will be 30x faster on CPU!\n")
    
    # Load minimal tasks (just 4 samples total)
    tasks, task_names = create_minimal_tasks()
    print(f"Tasks: {len(tasks)}")
    print(f"Total samples: {sum(len(t) for t in tasks)}")
    print(f"Task 1: {task_names[0]} - {len(tasks[0])} samples")
    print(f"Task 2: {task_names[1]} - {len(tasks[1])} samples")
    print()
    
    # CPU-optimized config with tiny model
    config = TrainingConfig(
        model_name="gpt2",  # Much smaller than Phi-3
        lora_r=4,           # Smaller LoRA rank
        lora_alpha=8,
        learning_rate=5e-4,
        max_guardrails=2,   # Fewer guardrails for speed
        enable_guardrails=True
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments") / f"mini_experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # ------------------------------------------------------------------
    # 1. Train SG-CL
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("  Training: SG-CL (with guardrails)")
    print("="*70 + "\n")
    
    config.enable_guardrails = True
    trainer_sgcl = SGCLTrainer(config)
    stats_sgcl = trainer_sgcl.train_on_tasks(tasks, task_names)
    
    sgcl_dir = output_dir / "sgcl_model"
    trainer_sgcl.save_model(str(sgcl_dir))
    
    results['sgcl'] = stats_sgcl
    
    # ------------------------------------------------------------------
    # 2. Train Baseline
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("  Training: Baseline (no guardrails)")
    print("="*70 + "\n")
    
    config.enable_guardrails = False
    trainer_baseline = NaiveFinetuningTrainer(config)
    stats_baseline = trainer_baseline.train_on_tasks(tasks, task_names)
    
    baseline_dir = output_dir / "baseline_model"
    trainer_baseline.save_model(str(baseline_dir))
    
    results['baseline'] = stats_baseline
    
    # ------------------------------------------------------------------
    # 3. Compare
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("  RESULTS COMPARISON")
    print("="*70)
    
    print("\n+- SG-CL (with guardrails) -------------------------------------+")
    print(f"| Conflicts Detected:  {stats_sgcl['conflicts_detected']:>3}                                      |")
    print(f"| Guardrails Added:    {stats_sgcl['total_guardrails']:>3}                                      |")
    print(f"| Conflict Rate:       {stats_sgcl['conflict_rate']:>5.1%}                                    |")
    print(f"| Avg Loss:            {stats_sgcl['avg_loss']:>6.4f}                                  |")
    print("+----------------------------------------------------------------+")
    
    print("\n+- Baseline (no guardrails) -------------------------------------+")
    print(f"| Conflicts Detected:  {stats_baseline['conflicts_detected']:>3}                                      |")
    print(f"| Guardrails Added:    {stats_baseline['total_guardrails']:>3}                                      |")
    print(f"| Conflict Rate:       {stats_baseline['conflict_rate']:>5.1%}                                    |")
    print(f"| Avg Loss:            {stats_baseline['avg_loss']:>6.4f}                                  |")
    print("+----------------------------------------------------------------+")
    
    # Save comparison
    comparison = {
        'experiment': 'mini_cpu_experiment',
        'timestamp': timestamp,
        'model': 'gpt2',
        'tasks': task_names,
        'results': results
    }
    
    comparison_file = output_dir / "comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✅ Experiment complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Comparison: {comparison_file}")
    
    return results

if __name__ == '__main__':
    results = run_mini_experiment()

"""
Complete SG-CL Training Script

Runs full training experiment comparing:
1. SG-CL (with SID + guardrails)
2. Naive fine-tuning baseline

This is the MAIN script for experiments.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from sgcl_training import SGCLTrainer, NaiveFinetuningTrainer, TrainingConfig
from sgcl_data_loader import load_seca_for_training, create_toy_tasks, create_minimal_tasks


def run_experiment(
    experiment_name: str,
    tasks,
    task_names,
    config: TrainingConfig,
    output_dir: str = "experiments"
):
    """
    Run complete SG-CL vs Baseline experiment.
    
    Args:
        experiment_name: Name for this experiment
        tasks: List of task data
        task_names: Task names
        config: Training configuration
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"  EXPERIMENT: {experiment_name}")
    print("="*80)
    print(f"Timestamp: {timestamp}")
    print(f"Output: {exp_dir}")
    print(f"Tasks: {len(tasks)}")
    print(f"Total Samples: {sum(len(t) for t in tasks)}")
    print("="*80 + "\n")
    
    results = {}
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Train SG-CL (with guardrails)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "▓"*80)
    print("  TRAINING: SG-CL (SID + Guardrails)")
    print("▓"*80 + "\n")
    
    sgcl_config = config
    sgcl_config.enable_guardrails = True
    
    sgcl_trainer = SGCLTrainer(sgcl_config)
    sgcl_stats = sgcl_trainer.train_on_tasks(tasks, task_names)
    
    sgcl_model_dir = exp_dir / "sgcl_model"
    sgcl_trainer.save_model(str(sgcl_model_dir))
    sgcl_trainer.save_statistics(str(exp_dir / "sgcl_stats.json"))
    
    results['sgcl'] = sgcl_stats
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Train Baseline (naive fine-tuning)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "▓"*80)
    print("  TRAINING: Baseline (Naive Fine-tuning)")
    print("▓"*80 + "\n")
    
    baseline_config = config
    baseline_config.enable_guardrails = False
    
    baseline_trainer = NaiveFinetuningTrainer(baseline_config)
    baseline_stats = baseline_trainer.train_on_tasks(tasks, task_names)
    
    baseline_model_dir = exp_dir / "baseline_model"
    baseline_trainer.save_model(str(baseline_model_dir))
    baseline_trainer.save_statistics(str(exp_dir / "baseline_stats.json"))
    
    results['baseline'] = baseline_stats
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Compare Results
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "="*80)
    print("  COMPARISON: SG-CL vs Baseline")
    print("="*80)
    
    print("\n┌─ SG-CL ─────────────────────────────────────────────────────────────┐")
    print(f"│ Conflict Rate:     {sgcl_stats['conflict_rate']:>6.1%}                                 │")
    print(f"│ Guardrails Added:  {sgcl_stats['total_guardrails']:>6}                                   │")
    print(f"│ Avg Loss:          {sgcl_stats['avg_loss']:>6.4f}                                 │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─ Baseline ──────────────────────────────────────────────────────────┐")
    print(f"│ Conflict Rate:     {baseline_stats['conflict_rate']:>6.1%}                                 │")
    print(f"│ Guardrails Added:  {baseline_stats['total_guardrails']:>6}                                   │")
    print(f"│ Avg Loss:          {baseline_stats['avg_loss']:>6.4f}                                 │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    
    # Save comparison
    comparison = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'config': {
            'model': config.model_name,
            'lora_r': config.lora_r,
            'learning_rate': config.learning_rate,
            'max_guardrails': config.max_guardrails
        },
        'tasks': {
            'count': len(tasks),
            'names': task_names,
            'total_samples': sum(len(t) for t in tasks)
        },
        'results': results
    }
    
    comparison_path = exp_dir / "comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Comparison saved: {comparison_path}")
    print(f"✓ Experiment complete: {exp_dir}")
    print("="*80 + "\n")
    
    return results


def main():
    """Main training script with CLI."""
    parser = argparse.ArgumentParser(
        description="SG-CL Training - Semantic-Guided Continual Learning"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['seca', 'toy', 'minimal'],
        default='minimal',
        help='Dataset to use (default: minimal for quick testing)'
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        default=None,
        help='Comma-separated task IDs to use (e.g., "0,1,2" for first 3 tasks)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='microsoft/phi-3-mini-4k-instruct',
        help='Model to use'
    )
    
    parser.add_argument(
        '--max-guardrails',
        type=int,
        default=4,
        help='Maximum guardrails per conflict'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='experiments',
        help='Output directory'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='sgcl_experiment',
        help='Experiment name'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    
    if args.dataset == 'seca':
        task_subset = None
        if args.tasks:
            task_subset = [int(x) for x in args.tasks.split(',')]
        tasks, task_names = load_seca_for_training(subset=task_subset)
    elif args.dataset == 'toy':
        tasks, task_names = create_toy_tasks()
    else:  # minimal
        tasks, task_names = create_minimal_tasks()
    
    print(f"✓ Loaded {len(tasks)} tasks with {sum(len(t) for t in tasks)} total samples")
    
    # Configure training
    config = TrainingConfig(
        model_name=args.model,
        max_guardrails=args.max_guardrails,
        learning_rate=args.lr
    )
    
    # Run experiment
    results = run_experiment(
        experiment_name=args.name,
        tasks=tasks,
        task_names=task_names,
        config=config,
        output_dir=args.output
    )
    
    print("\n✅ Experiment Complete!")


if __name__ == '__main__':
    main()

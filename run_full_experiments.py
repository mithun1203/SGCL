"""
Full SG-CL Experiments Runner for Kaggle

Runs complete experiments comparing:
- SG-CL (with SID + guardrails)
- Naive Fine-Tuning
- EWC
- Experience Replay

On full SeCA v2.0 dataset (320 samples, 8 tasks).
Collects all metrics for publication.
"""

import torch
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys

# Import training methods
from sgcl_training import SGCLTrainer, TrainingConfig
from baseline_methods import (
    NaiveFinetuningTrainer,
    EWCTrainer,
    ReplayTrainer,
    BaselineConfig
)
from scp_evaluation import compare_methods
from sgcl_data_loader import load_seca_tasks


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_full_experiments(
    model_name: str = "microsoft/phi-3-mini-4k-instruct",
    use_mini_dataset: bool = False,
    output_dir: str = "experiments",
    lora_r: int = 8,
    lora_alpha: int = 16,
    max_steps_per_task: Optional[int] = None
):
    """
    Run complete experiments on SeCA dataset.
    
    Args:
        model_name: Base model to use
        use_mini_dataset: If True, use mini dataset for testing (faster)
        output_dir: Where to save results
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        max_steps_per_task: Limit samples per task (None = use all)
    """
    print_header("SG-CL FULL EXPERIMENTS")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(output_dir) / f"full_experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output Directory: {experiment_dir}")
    print(f"🤖 Model: {model_name}")
    print(f"💾 Mini Dataset: {use_mini_dataset}")
    print(f"🔧 LoRA Config: r={lora_r}, alpha={lora_alpha}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  Running on CPU (this will be SLOW!)")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: Load Data
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_header("PHASE 1: Loading SeCA Dataset")
    
    if use_mini_dataset:
        from sgcl_data_loader import create_minimal_tasks
        tasks, task_names = create_minimal_tasks()
        print(f"✓ Loaded mini dataset: {len(tasks)} tasks")
    else:
        tasks, task_names = load_seca_tasks()
        print(f"✓ Loaded full SeCA v2.0: {len(tasks)} tasks")
    
    # Limit samples per task if specified
    if max_steps_per_task:
        tasks = [task[:max_steps_per_task] for task in tasks]
        print(f"✓ Limited to {max_steps_per_task} samples per task")
    
    for i, (task, name) in enumerate(zip(tasks, task_names)):
        print(f"  Task {i+1}: {name:30} | {len(task):3} samples")
    
    total_samples = sum(len(task) for task in tasks)
    print(f"\n📊 Total samples: {total_samples}")
    
    # Split into train/test (80/20)
    train_tasks = []
    test_tasks = []
    for task in tasks:
        split_idx = int(len(task) * 0.8)
        train_tasks.append(task[:split_idx])
        test_tasks.append(task[split_idx:])
    
    print(f"✓ Train samples: {sum(len(t) for t in train_tasks)}")
    print(f"✓ Test samples: {sum(len(t) for t in test_tasks)}")
    
    # Save dataset info
    dataset_info = {
        'model_name': model_name,
        'num_tasks': len(tasks),
        'task_names': task_names,
        'total_samples': total_samples,
        'train_samples': sum(len(t) for t in train_tasks),
        'test_samples': sum(len(t) for t in test_tasks),
        'timestamp': timestamp
    }
    
    with open(experiment_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: Train All Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    methods_to_run = {
        'sgcl': 'SG-CL (Ours)',
        'naive': 'Naive Fine-Tuning',
        'ewc': 'EWC',
        'replay': 'Experience Replay'
    }
    
    training_results = {}
    model_paths = {}
    
    for method_key, method_name in methods_to_run.items():
        print_header(f"PHASE 2: Training {method_name}")
        
        method_dir = experiment_dir / method_key
        method_dir.mkdir(exist_ok=True)
        
        try:
            # Configure method
            if method_key == 'sgcl':
                config = TrainingConfig(
                    model_name=model_name,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    enable_guardrails=True,
                    max_guardrails=4
                )
                trainer = SGCLTrainer(config)
            
            elif method_key == 'naive':
                config = BaselineConfig(
                    model_name=model_name,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha
                )
                trainer = NaiveFinetuningTrainer(config)
            
            elif method_key == 'ewc':
                config = BaselineConfig(
                    model_name=model_name,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    ewc_lambda=5000.0
                )
                trainer = EWCTrainer(config)
            
            elif method_key == 'replay':
                config = BaselineConfig(
                    model_name=model_name,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    replay_buffer_size=100,
                    replay_batch_size=2
                )
                trainer = ReplayTrainer(config)
            
            # Train
            stats = trainer.train_on_tasks(train_tasks, task_names)
            
            # Save model
            trainer.save_model(str(method_dir / "model"))
            
            # Save training stats
            with open(method_dir / "training_stats.json", 'w') as f:
                json.dump(stats, f, indent=2)
            
            training_results[method_key] = stats
            model_paths[method_key] = str(method_dir / "model")
            
            print(f"✓ {method_name} training completed")
            print(f"✓ Model saved to: {method_dir / 'model'}")
            
        except Exception as e:
            print(f"✗ Error training {method_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: Evaluate All Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_header("PHASE 3: Evaluating All Methods")
    
    evaluation_results = {}
    
    try:
        evaluation_results = compare_methods(
            model_paths=model_paths,
            test_tasks=test_tasks,
            task_names=task_names,
            output_dir=str(experiment_dir / "evaluation")
        )
        
        print("✓ Evaluation completed for all methods")
        
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 4: Compile Final Results
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_header("PHASE 4: Compiling Final Results")
    
    final_results = {
        'experiment_info': dataset_info,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'summary': {}
    }
    
    # Create summary table
    summary_table = []
    for method_key in model_paths.keys():
        if method_key in evaluation_results:
            eval_data = evaluation_results[method_key]
            metrics = eval_data.get('metrics', {})
            
            summary_table.append({
                'method': methods_to_run[method_key],
                'overall_score': eval_data.get('overall_score', 0.0),
                'semantic_consistency': metrics.get('semantic_consistency', 0.0),
                'contradiction_rate': metrics.get('contradiction_rate', 0.0),
                'avg_forgetting': metrics.get('forgetting', {}).get('avg_forgetting', 0.0),
                'avg_accuracy': metrics.get('task_accuracy', {}).get('avg_accuracy', 0.0)
            })
    
    final_results['summary']['comparison_table'] = summary_table
    
    # Save final results
    results_file = experiment_dir / "final_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✓ Final results saved to: {results_file}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 5: Print Summary
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_header("FINAL RESULTS SUMMARY")
    
    print(f"\n{'Method':<25} | {'Overall':>8} | {'Consistency':>12} | {'Contradiction':>14} | {'Forgetting':>11} | {'Accuracy':>9}")
    print("-" * 105)
    
    for row in summary_table:
        print(f"{row['method']:<25} | "
              f"{row['overall_score']:>8.4f} | "
              f"{row['semantic_consistency']:>12.4f} | "
              f"{row['contradiction_rate']:>14.4f} | "
              f"{row['avg_forgetting']:>11.4f} | "
              f"{row['avg_accuracy']:>9.4f}")
    
    print("\n" + "="*80)
    print(f"✅ EXPERIMENT COMPLETED!")
    print(f"📁 Results saved to: {experiment_dir}")
    print("="*80 + "\n")
    
    return final_results


def run_quick_comparison(
    model_name: str = "gpt2",
    output_dir: str = "experiments"
):
    """
    Quick comparison on mini dataset (for testing).
    
    Uses GPT-2 for speed, mini dataset with 4 samples.
    Perfect for verifying everything works before full run.
    """
    print_header("QUICK COMPARISON TEST")
    print("Using mini dataset + GPT-2 for fast verification\n")
    
    return run_full_experiments(
        model_name=model_name,
        use_mini_dataset=True,
        output_dir=output_dir,
        lora_r=4,
        lora_alpha=8,
        max_steps_per_task=2  # Only 2 samples per task
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SG-CL experiments')
    parser.add_argument('--model', type=str, default='microsoft/phi-3-mini-4k-instruct',
                       help='Model name')
    parser.add_argument('--mini', action='store_true',
                       help='Use mini dataset for testing')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with GPT-2 + mini dataset')
    parser.add_argument('--output', type=str, default='experiments',
                       help='Output directory')
    parser.add_argument('--lora-r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Max samples per task (None = all)')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Running QUICK comparison (GPT-2 + mini dataset)")
        run_quick_comparison(output_dir=args.output)
    else:
        print("🚀 Running FULL experiments")
        run_full_experiments(
            model_name=args.model,
            use_mini_dataset=args.mini,
            output_dir=args.output,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            max_steps_per_task=args.max_steps
        )

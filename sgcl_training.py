"""
SG-CL Training Integration

Connects SID + Guardrails + LoRA training into a complete continual learning system.
This is the CORE training loop that makes SG-CL work.

Architecture:
    SeCA Task → SID Conflict Check → IF conflict THEN add guardrails → Train
"""

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from tqdm import tqdm

from sid import SemanticInconsistencyDetector
from guardrail import GuardrailController


@dataclass
class TrainingConfig:
    """Configuration for SG-CL training."""
    model_name: str = "microsoft/phi-3-mini-4k-instruct"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 1  # Process one sample at a time for SID gating
    max_guardrails: int = 4
    enable_guardrails: bool = True  # Set False for baseline
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingMetrics:
    """Metrics logged during training."""
    task_id: int
    sample_idx: int
    loss: float
    conflict_detected: bool
    guardrails_added: int
    total_samples_in_batch: int


class SGCLTrainer:
    """
    SG-CL Training Integration.
    
    This is the core training loop that implements:
    1. Sequential task training (continual learning)
    2. SID-based conflict detection
    3. Guard-rail augmentation when conflicts detected
    4. Standard gradient updates (no special optimizer)
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        sid_kb_path: Optional[str] = None
    ):
        """
        Initialize SG-CL trainer.
        
        Args:
            config: Training configuration
            sid_kb_path: Path to SID knowledge base (optional)
        """
        self.config = config
        
        # Initialize model + tokenizer
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=config.lora_dropout,
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA applied: r={config.lora_r}, alpha={config.lora_alpha}")
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize SID + Guardrail Controller
        self.guardrail_controller = GuardrailController(
            max_guardrails=config.max_guardrails,
            enable_guardrails=config.enable_guardrails
        )
        
        # Metrics storage
        self.metrics: List[TrainingMetrics] = []
        self.knowledge_base: List[str] = []  # Accumulates learned knowledge
    
    def train_on_tasks(
        self,
        tasks: List[List[str]],
        task_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train on sequential tasks (continual learning).
        
        THIS IS THE CORE SG-CL TRAINING LOOP.
        
        Args:
            tasks: List of tasks, each task is List[str] of training samples
            task_names: Optional names for tasks
        
        Returns:
            Training statistics
        """
        if task_names is None:
            task_names = [f"Task_{i}" for i in range(len(tasks))]
        
        print("\n" + "="*70)
        print("  SG-CL Training Started")
        print("="*70)
        print(f"Model: {self.config.model_name}")
        print(f"Tasks: {len(tasks)}")
        print(f"Guardrails: {'ENABLED' if self.config.enable_guardrails else 'DISABLED'}")
        print(f"Max Guardrails per Conflict: {self.config.max_guardrails}")
        print("="*70 + "\n")
        
        for task_id, (task_data, task_name) in enumerate(zip(tasks, task_names)):
            print(f"\n{'─'*70}")
            print(f"Training on {task_name} (Task {task_id + 1}/{len(tasks)})")
            print(f"Samples: {len(task_data)}")
            print(f"{'─'*70}")
            
            self._train_on_task(task_id, task_data)
            
            # Add task data to knowledge base for future conflict detection
            self.knowledge_base.extend(task_data)
        
        # Compute statistics
        stats = self._compute_statistics()
        
        print("\n" + "="*70)
        print("  SG-CL Training Completed")
        print("="*70)
        self._print_statistics(stats)
        
        return stats
    
    def _train_on_task(self, task_id: int, task_data: List[str]):
        """
        Train on a single task.
        
        For each sample:
        1. Check semantic conflict with SID
        2. If conflict → add guardrails
        3. Train on final batch (sample + guardrails)
        """
        self.model.train()
        
        for sample_idx, sample in enumerate(tqdm(task_data, desc="Training")):
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # CORE SG-CL LOGIC (THIS IS WHAT MAKES IT SG-CL)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            # Step 1: Check semantic conflict
            result = self.guardrail_controller.process_batch(
                [sample],
                self.knowledge_base
            )
            
            # Step 2: Construct training batch
            if result.has_conflict and self.config.enable_guardrails:
                # Conflict detected → augment with guardrails
                training_batch = result.original_samples + result.guardrail_samples
                conflict_detected = True
                guardrails_added = len(result.guardrail_samples)
            else:
                # No conflict → train normally
                training_batch = [sample]
                conflict_detected = False
                guardrails_added = 0
            
            # Step 3: Train on batch
            loss = self._train_step(training_batch)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            # Log metrics
            self.metrics.append(TrainingMetrics(
                task_id=task_id,
                sample_idx=sample_idx,
                loss=loss,
                conflict_detected=conflict_detected,
                guardrails_added=guardrails_added,
                total_samples_in_batch=len(training_batch)
            ))
    
    def _train_step(self, batch: List[str]) -> float:
        """
        Single training step (standard gradient update).
        
        NO SPECIAL OPTIMIZER. NO LOSS MODIFICATION.
        Just normal fine-tuning on the (possibly augmented) batch.
        """
        # Tokenize
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Forward pass
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute training statistics."""
        total_samples = len(self.metrics)
        conflicts_detected = sum(1 for m in self.metrics if m.conflict_detected)
        total_guardrails = sum(m.guardrails_added for m in self.metrics)
        avg_loss = sum(m.loss for m in self.metrics) / total_samples if total_samples > 0 else 0
        
        # Per-task statistics
        task_stats = {}
        for task_id in range(max(m.task_id for m in self.metrics) + 1):
            task_metrics = [m for m in self.metrics if m.task_id == task_id]
            task_conflicts = sum(1 for m in task_metrics if m.conflict_detected)
            task_guardrails = sum(m.guardrails_added for m in task_metrics)
            
            task_stats[f"task_{task_id}"] = {
                "samples": len(task_metrics),
                "conflicts": task_conflicts,
                "guardrails": task_guardrails,
                "conflict_rate": task_conflicts / len(task_metrics) if task_metrics else 0,
                "avg_loss": sum(m.loss for m in task_metrics) / len(task_metrics) if task_metrics else 0
            }
        
        return {
            "total_samples": total_samples,
            "conflicts_detected": conflicts_detected,
            "conflict_rate": conflicts_detected / total_samples if total_samples > 0 else 0,
            "total_guardrails": total_guardrails,
            "avg_guardrails_per_conflict": total_guardrails / conflicts_detected if conflicts_detected > 0 else 0,
            "avg_loss": avg_loss,
            "task_stats": task_stats
        }
    
    def _print_statistics(self, stats: Dict[str, Any]):
        """Print training statistics."""
        print(f"\nTotal Samples Trained: {stats['total_samples']}")
        print(f"Conflicts Detected: {stats['conflicts_detected']}")
        print(f"Conflict Rate: {stats['conflict_rate']:.2%}")
        print(f"Total Guardrails Injected: {stats['total_guardrails']}")
        print(f"Avg Guardrails per Conflict: {stats['avg_guardrails_per_conflict']:.1f}")
        print(f"Average Loss: {stats['avg_loss']:.4f}")
        
        print("\nPer-Task Statistics:")
        for task_name, task_stat in stats['task_stats'].items():
            print(f"\n  {task_name}:")
            print(f"    Samples: {task_stat['samples']}")
            print(f"    Conflicts: {task_stat['conflicts']} ({task_stat['conflict_rate']:.1%})")
            print(f"    Guardrails: {task_stat['guardrails']}")
            print(f"    Avg Loss: {task_stat['avg_loss']:.4f}")
    
    def save_model(self, output_dir: str):
        """Save trained model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save metrics
        metrics_path = output_path / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump([
                {
                    'task_id': m.task_id,
                    'sample_idx': m.sample_idx,
                    'loss': m.loss,
                    'conflict_detected': m.conflict_detected,
                    'guardrails_added': m.guardrails_added
                }
                for m in self.metrics
            ], f, indent=2)
        
        print(f"\nModel saved to: {output_path}")
        print(f"Metrics saved to: {metrics_path}")
    
    def save_statistics(self, output_path: str):
        """Save training statistics to file."""
        stats = self._compute_statistics()
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Baseline Trainers (for comparison)
# ══════════════════════════════════════════════════════════════════════════════

class NaiveFinetuningTrainer(SGCLTrainer):
    """
    Baseline: Naive Fine-tuning (no SID, no guardrails).
    
    Same model, same optimizer, but NO semantic intervention.
    """
    
    def __init__(self, config: TrainingConfig):
        # Disable guardrails for baseline
        config.enable_guardrails = False
        super().__init__(config)
        print("Baseline: Naive Fine-tuning (no SID, no guardrails)")


# ══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ══════════════════════════════════════════════════════════════════════════════

def train_sgcl(
    tasks: List[List[str]],
    output_dir: str = "models/sgcl",
    **config_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train SG-CL model.
    
    Args:
        tasks: List of tasks (each task is list of training samples)
        output_dir: Where to save model
        **config_kwargs: Override default config
    
    Returns:
        Training statistics
    """
    config = TrainingConfig(**config_kwargs)
    trainer = SGCLTrainer(config)
    stats = trainer.train_on_tasks(tasks)
    trainer.save_model(output_dir)
    
    return stats


def train_baseline(
    tasks: List[List[str]],
    output_dir: str = "models/baseline",
    **config_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train baseline (naive fine-tuning).
    
    Args:
        tasks: List of tasks
        output_dir: Where to save model
        **config_kwargs: Override default config
    
    Returns:
        Training statistics
    """
    config = TrainingConfig(**config_kwargs)
    trainer = NaiveFinetuningTrainer(config)
    stats = trainer.train_on_tasks(tasks)
    trainer.save_model(output_dir)
    
    return stats


if __name__ == '__main__':
    # Demo with toy tasks
    print("SG-CL Training Integration Demo")
    print("="*70)
    
    # Create toy sequential tasks
    tasks = [
        [
            "Eagles have sharp talons.",
            "Birds can fly.",
            "Sparrows are small birds."
        ],
        [
            "Penguins cannot fly.",  # Conflicts with Task 1
            "Penguins are birds.",
            "Penguins live in Antarctica."
        ]
    ]
    
    task_names = ["Bird Facts (General)", "Penguin Facts (Exception)"]
    
    # Train SG-CL
    config = TrainingConfig(
        model_name="microsoft/phi-3-mini-4k-instruct",
        enable_guardrails=True,
        max_guardrails=4
    )
    
    trainer = SGCLTrainer(config)
    stats = trainer.train_on_tasks(tasks, task_names)
    
    # Save
    trainer.save_model("models/sgcl_demo")
    trainer.save_statistics("results/sgcl_stats.json")

"""
Baseline Continual Learning Methods

Implements comparison baselines:
1. Naive Fine-Tuning (sequential training, no mitigation)
2. EWC (Elastic Weight Consolidation)
3. Experience Replay (memory buffer)

All use same model architecture (Phi-3 + LoRA) for fair comparison.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm
import random
import copy


@dataclass
class BaselineConfig:
    """Configuration for baseline methods."""
    model_name: str = "microsoft/phi-3-mini-4k-instruct"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # EWC specific
    ewc_lambda: float = 5000.0  # Importance of old tasks
    
    # Replay specific
    replay_buffer_size: int = 100  # Samples to store per task
    replay_batch_size: int = 2  # How many replay samples per training step


class BaselineTrainer:
    """Base class for all baseline methods."""
    
    def __init__(self, config: BaselineConfig):
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
        if "gpt2" in config.model_name.lower():
            target_modules = ["c_attn"]
        else:
            target_modules = ["q_proj", "v_proj"]
            
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Metrics storage
        self.metrics = []
    
    def train_on_tasks(
        self,
        tasks: List[List[str]],
        task_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train on sequential tasks."""
        raise NotImplementedError("Subclass must implement")
    
    def _train_step(self, batch: List[str]) -> float:
        """Single training step."""
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def save_model(self, output_dir: str):
        """Save trained model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save metrics
        metrics_path = output_path / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Model saved to: {output_path}")


# ==============================================================================
# Method 1: Naive Fine-Tuning
# ==============================================================================

class NaiveFinetuningTrainer(BaselineTrainer):
    """
    Baseline 1: Naive Sequential Fine-Tuning
    
    Simply trains on tasks sequentially with no catastrophic forgetting mitigation.
    This is the WORST baseline - shows what happens without any intervention.
    """
    
    def train_on_tasks(
        self,
        tasks: List[List[str]],
        task_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        
        if task_names is None:
            task_names = [f"Task_{i}" for i in range(len(tasks))]
        
        print("\n" + "="*70)
        print("  Naive Fine-Tuning Training")
        print("="*70)
        print(f"Model: {self.config.model_name}")
        print(f"Tasks: {len(tasks)}")
        print("WARNING: No catastrophic forgetting mitigation!")
        print("="*70 + "\n")
        
        for task_id, (task_data, task_name) in enumerate(zip(tasks, task_names)):
            print(f"\n{'-'*70}")
            print(f"Training on {task_name} (Task {task_id + 1}/{len(tasks)})")
            print(f"Samples: {len(task_data)}")
            print(f"{'='*70}")
            
            self.model.train()
            
            for sample_idx, sample in enumerate(tqdm(task_data, desc="Training")):
                loss = self._train_step([sample])
                
                self.metrics.append({
                    'task_id': task_id,
                    'sample_idx': sample_idx,
                    'loss': loss,
                    'method': 'naive'
                })
        
        # Compute statistics
        stats = self._compute_statistics()
        
        print("\n" + "="*70)
        print("  Naive Fine-Tuning Completed")
        print("="*70)
        self._print_statistics(stats)
        
        return stats
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute training statistics."""
        total_samples = len(self.metrics)
        avg_loss = sum(m['loss'] for m in self.metrics) / total_samples if total_samples > 0 else 0
        
        # Per-task statistics
        task_stats = {}
        for task_id in range(max(m['task_id'] for m in self.metrics) + 1):
            task_metrics = [m for m in self.metrics if m['task_id'] == task_id]
            
            task_stats[f"task_{task_id}"] = {
                "samples": len(task_metrics),
                "avg_loss": sum(m['loss'] for m in task_metrics) / len(task_metrics) if task_metrics else 0
            }
        
        return {
            "method": "Naive Fine-Tuning",
            "total_samples": total_samples,
            "avg_loss": avg_loss,
            "task_stats": task_stats
        }
    
    def _print_statistics(self, stats: Dict[str, Any]):
        """Print training statistics."""
        print(f"\nTotal Samples Trained: {stats['total_samples']}")
        print(f"Average Loss: {stats['avg_loss']:.4f}")
        
        print("\nPer-Task Statistics:")
        for task_name, task_stat in stats['task_stats'].items():
            print(f"\n  {task_name}:")
            print(f"    Samples: {task_stat['samples']}")
            print(f"    Avg Loss: {task_stat['avg_loss']:.4f}")


# ==============================================================================
# Method 2: EWC (Elastic Weight Consolidation)
# ==============================================================================

class EWCTrainer(BaselineTrainer):
    """
    Baseline 2: Elastic Weight Consolidation (EWC)
    
    Penalizes changes to important parameters from previous tasks.
    Uses Fisher Information Matrix to estimate parameter importance.
    
    Reference: Kirkpatrick et al. (2017) - "Overcoming catastrophic forgetting"
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__(config)
        
        # EWC specific: store important parameters after each task
        self.fisher_dict = {}  # Fisher information matrices
        self.optpar_dict = {}  # Optimal parameters from previous tasks
        
        print(f"EWC lambda: {config.ewc_lambda}")
    
    def train_on_tasks(
        self,
        tasks: List[List[str]],
        task_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        
        if task_names is None:
            task_names = [f"Task_{i}" for i in range(len(tasks))]
        
        print("\n" + "="*70)
        print("  EWC Training")
        print("="*70)
        print(f"Model: {self.config.model_name}")
        print(f"Tasks: {len(tasks)}")
        print(f"EWC Lambda: {self.config.ewc_lambda}")
        print("="*70 + "\n")
        
        for task_id, (task_data, task_name) in enumerate(zip(tasks, task_names)):
            print(f"\n{'-'*70}")
            print(f"Training on {task_name} (Task {task_id + 1}/{len(tasks)})")
            print(f"Samples: {len(task_data)}")
            print(f"{'='*70}")
            
            self.model.train()
            
            for sample_idx, sample in enumerate(tqdm(task_data, desc="Training")):
                loss = self._train_step_ewc([sample], task_id)
                
                self.metrics.append({
                    'task_id': task_id,
                    'sample_idx': sample_idx,
                    'loss': loss,
                    'method': 'ewc'
                })
            
            # After task completes, compute Fisher Information
            print("Computing Fisher Information...")
            self._compute_fisher(task_data, task_id)
        
        # Compute statistics
        stats = self._compute_statistics()
        
        print("\n" + "="*70)
        print("  EWC Training Completed")
        print("="*70)
        self._print_statistics(stats)
        
        return stats
    
    def _train_step_ewc(self, batch: List[str], current_task_id: int) -> float:
        """Training step with EWC penalty."""
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Standard loss
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # EWC penalty: penalize deviating from important parameters
        if current_task_id > 0:
            ewc_loss = 0.0
            for name, param in self.model.named_parameters():
                if name in self.fisher_dict:
                    fisher = self.fisher_dict[name]
                    optpar = self.optpar_dict[name]
                    ewc_loss += (fisher * (param - optpar).pow(2)).sum()
            
            loss = loss + self.config.ewc_lambda * ewc_loss
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def _compute_fisher(self, task_data: List[str], task_id: int):
        """
        Compute Fisher Information Matrix for current task.
        
        Estimates parameter importance based on gradient magnitudes.
        """
        self.model.eval()
        
        # Initialize Fisher dict for this task
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        # Accumulate gradients
        for sample in task_data[:20]:  # Use subset for efficiency
            inputs = self.tokenizer(
                [sample],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher approximation)
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) / len(task_data[:20])
        
        # Store Fisher information and current parameters
        for n, p in self.model.named_parameters():
            if n in fisher:
                # Accumulate Fisher from previous tasks
                if n in self.fisher_dict:
                    self.fisher_dict[n] = self.fisher_dict[n] + fisher[n]
                else:
                    self.fisher_dict[n] = fisher[n]
                
                # Store optimal parameters
                self.optpar_dict[n] = p.data.clone()
        
        self.model.train()
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute training statistics."""
        total_samples = len(self.metrics)
        avg_loss = sum(m['loss'] for m in self.metrics) / total_samples if total_samples > 0 else 0
        
        task_stats = {}
        for task_id in range(max(m['task_id'] for m in self.metrics) + 1):
            task_metrics = [m for m in self.metrics if m['task_id'] == task_id]
            
            task_stats[f"task_{task_id}"] = {
                "samples": len(task_metrics),
                "avg_loss": sum(m['loss'] for m in task_metrics) / len(task_metrics) if task_metrics else 0
            }
        
        return {
            "method": "EWC",
            "total_samples": total_samples,
            "avg_loss": avg_loss,
            "ewc_lambda": self.config.ewc_lambda,
            "task_stats": task_stats
        }
    
    def _print_statistics(self, stats: Dict[str, Any]):
        """Print training statistics."""
        print(f"\nTotal Samples Trained: {stats['total_samples']}")
        print(f"Average Loss: {stats['avg_loss']:.4f}")
        print(f"EWC Lambda: {stats['ewc_lambda']}")
        
        print("\nPer-Task Statistics:")
        for task_name, task_stat in stats['task_stats'].items():
            print(f"\n  {task_name}:")
            print(f"    Samples: {task_stat['samples']}")
            print(f"    Avg Loss: {task_stat['avg_loss']:.4f}")


# ==============================================================================
# Method 3: Experience Replay
# ==============================================================================

class ReplayTrainer(BaselineTrainer):
    """
    Baseline 3: Experience Replay
    
    Stores samples from previous tasks in a memory buffer.
    Interleaves replay samples with current task samples during training.
    
    Reference: Rolnick et al. (2019) - "Experience Replay for CL"
    """
    
    def __init__(self, config: BaselineConfig):
        super().__init__(config)
        
        # Replay buffer: stores samples from previous tasks
        self.replay_buffer: List[str] = []
        
        print(f"Replay buffer size: {config.replay_buffer_size}")
        print(f"Replay batch size: {config.replay_batch_size}")
    
    def train_on_tasks(
        self,
        tasks: List[List[str]],
        task_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        
        if task_names is None:
            task_names = [f"Task_{i}" for i in range(len(tasks))]
        
        print("\n" + "="*70)
        print("  Experience Replay Training")
        print("="*70)
        print(f"Model: {self.config.model_name}")
        print(f"Tasks: {len(tasks)}")
        print(f"Buffer Size: {self.config.replay_buffer_size}")
        print("="*70 + "\n")
        
        for task_id, (task_data, task_name) in enumerate(zip(tasks, task_names)):
            print(f"\n{'-'*70}")
            print(f"Training on {task_name} (Task {task_id + 1}/{len(tasks)})")
            print(f"Samples: {len(task_data)}")
            print(f"Replay Buffer Size: {len(self.replay_buffer)}")
            print(f"{'='*70}")
            
            self.model.train()
            
            for sample_idx, sample in enumerate(tqdm(task_data, desc="Training")):
                # Construct batch: current sample + replay samples
                batch = [sample]
                
                # Add replay samples if buffer is non-empty
                if len(self.replay_buffer) > 0:
                    n_replay = min(self.config.replay_batch_size, len(self.replay_buffer))
                    replay_samples = random.sample(self.replay_buffer, n_replay)
                    batch.extend(replay_samples)
                
                # Train on combined batch
                loss = self._train_step(batch)
                
                self.metrics.append({
                    'task_id': task_id,
                    'sample_idx': sample_idx,
                    'loss': loss,
                    'replay_samples': len(batch) - 1,
                    'method': 'replay'
                })
            
            # After task completes, add samples to replay buffer
            self._update_replay_buffer(task_data)
        
        # Compute statistics
        stats = self._compute_statistics()
        
        print("\n" + "="*70)
        print("  Experience Replay Training Completed")
        print("="*70)
        self._print_statistics(stats)
        
        return stats
    
    def _update_replay_buffer(self, task_data: List[str]):
        """Add task samples to replay buffer (reservoir sampling)."""
        # Simple strategy: keep up to buffer_size random samples per task
        samples_to_add = min(
            self.config.replay_buffer_size // 10,  # Reserve space for multiple tasks
            len(task_data)
        )
        
        sampled = random.sample(task_data, samples_to_add)
        self.replay_buffer.extend(sampled)
        
        # Prune buffer if too large
        if len(self.replay_buffer) > self.config.replay_buffer_size:
            self.replay_buffer = random.sample(self.replay_buffer, self.config.replay_buffer_size)
        
        print(f"Updated replay buffer: {len(self.replay_buffer)} samples")
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute training statistics."""
        total_samples = len(self.metrics)
        avg_loss = sum(m['loss'] for m in self.metrics) / total_samples if total_samples > 0 else 0
        total_replay_samples = sum(m.get('replay_samples', 0) for m in self.metrics)
        
        task_stats = {}
        for task_id in range(max(m['task_id'] for m in self.metrics) + 1):
            task_metrics = [m for m in self.metrics if m['task_id'] == task_id]
            task_replay = sum(m.get('replay_samples', 0) for m in task_metrics)
            
            task_stats[f"task_{task_id}"] = {
                "samples": len(task_metrics),
                "replay_samples": task_replay,
                "avg_loss": sum(m['loss'] for m in task_metrics) / len(task_metrics) if task_metrics else 0
            }
        
        return {
            "method": "Experience Replay",
            "total_samples": total_samples,
            "total_replay_samples": total_replay_samples,
            "avg_loss": avg_loss,
            "final_buffer_size": len(self.replay_buffer),
            "task_stats": task_stats
        }
    
    def _print_statistics(self, stats: Dict[str, Any]):
        """Print training statistics."""
        print(f"\nTotal Samples Trained: {stats['total_samples']}")
        print(f"Total Replay Samples Used: {stats['total_replay_samples']}")
        print(f"Final Buffer Size: {stats['final_buffer_size']}")
        print(f"Average Loss: {stats['avg_loss']:.4f}")
        
        print("\nPer-Task Statistics:")
        for task_name, task_stat in stats['task_stats'].items():
            print(f"\n  {task_name}:")
            print(f"    Samples: {task_stat['samples']}")
            print(f"    Replay Samples: {task_stat['replay_samples']}")
            print(f"    Avg Loss: {task_stat['avg_loss']:.4f}")

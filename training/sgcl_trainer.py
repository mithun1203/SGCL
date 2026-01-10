"""
SG-CL Training Engine - Main Training Loop
==========================================

Implements the core SG-CL algorithm:
1. Load batch from SeCA dataset
2. SID checks for conflicts
3. If conflict → Generate guard-rails
4. Augment batch with guard-rails
5. Forward pass + Loss computation
6. Backpropagation + Update

Author: Mithun Naik
Project: SGCL Capstone
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import time

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sid.detector import SemanticInconsistencyDetector
from guardrail.guardrail_generator import GuardrailGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for SG-CL training."""
    # Model settings
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    max_length: int = 512
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Training settings
    batch_size: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # SG-CL specific settings
    enable_sid: bool = True
    enable_guardrails: bool = True
    max_guardrails_per_conflict: int = 4
    
    # Logging & checkpointing
    log_every_n_steps: int = 10
    save_every_n_steps: int = 500
    output_dir: str = "./sgcl_checkpoints"
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default LoRA targets for Phi-3
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class SeCADataset(Dataset):
    """Dataset wrapper for SeCA 10K."""
    
    def __init__(self, data_path: str, task_id: Optional[int] = None):
        """
        Initialize SeCA dataset.
        
        Args:
            data_path: Path to seca_10k_final.json
            task_id: If specified, only load samples from this task
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract samples
        self.samples = []
        tasks = data.get('tasks', [])
        
        for task in tasks:
            if task_id is not None and task.get('task_id') != task_id:
                continue
            self.samples.extend(task.get('samples', []))
        
        logger.info(f"Loaded {len(self.samples)} samples" + 
                   (f" from task {task_id}" if task_id is not None else ""))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class SGCLTrainer:
    """
    SG-CL Training Engine.
    
    Implements the complete SG-CL algorithm with:
    - SID conflict detection
    - Guard-rail generation
    - LoRA fine-tuning
    - Task-incremental learning
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize SG-CL trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🚀 Initializing SG-CL Trainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {config.model_name}")
        
        # Initialize components
        self._setup_model()
        self._setup_sid_and_guardrails()
        
        # Training state
        self.global_step = 0
        self.current_task = 0
        self.training_metrics = {
            'loss': [],
            'conflicts_detected': 0,
            'guardrails_generated': 0,
            'samples_processed': 0
        }
    
    def _setup_model(self):
        """Load model and attach LoRA adapters."""
        logger.info("📦 Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)
        
        logger.info(f"✓ Model loaded with LoRA adapters")
        self.model.print_trainable_parameters()
    
    def _setup_sid_and_guardrails(self):
        """Initialize SID and guard-rail generator."""
        if self.config.enable_sid:
            logger.info("🧠 Initializing SID...")
            self.sid = SemanticInconsistencyDetector()
            logger.info("✓ SID initialized")
        else:
            self.sid = None
        
        if self.config.enable_guardrails:
            logger.info("🛡️ Initializing Guard-Rail Generator...")
            self.guardrail_gen = GuardrailGenerator()
            logger.info("✓ Guard-Rail Generator initialized")
        else:
            self.guardrail_gen = None
    
    def _check_conflicts_and_generate_guardrails(
        self, 
        texts: List[str]
    ) -> Tuple[List[str], int, int]:
        """
        Apply SG-CL: Check conflicts and generate guard-rails.
        
        Args:
            texts: Batch of input texts
        
        Returns:
            Tuple of (augmented_texts, conflicts_detected, guardrails_generated)
        """
        if not self.config.enable_sid:
            return texts, 0, 0
        
        augmented_texts = []
        conflicts_detected = 0
        guardrails_generated = 0
        
        for text in texts:
            # Always add original text
            augmented_texts.append(text)
            
            # Check for conflicts
            result = self.sid.detect_conflict(text)
            
            if result.has_conflict and self.config.enable_guardrails:
                conflicts_detected += 1
                
                # Generate guard-rails for each conflict
                for conflict in result.conflicts:
                    try:
                        entity = conflict.source_triple.subject
                        relation = conflict.source_triple.relation.replace("Not", "")
                        obj = conflict.source_triple.object
                        
                        rails = self.guardrail_gen.generate(
                            conflict_entity=entity,
                            conflict_relation=relation,
                            conflict_object=obj,
                            max_facts=self.config.max_guardrails_per_conflict
                        )
                        
                        # Add guard-rail sentences to batch
                        for rail in rails:
                            augmented_texts.append(rail.sentence)
                            guardrails_generated += 1
                    
                    except Exception as e:
                        logger.warning(f"Failed to generate guard-rails: {e}")
        
        return augmented_texts, conflicts_detected, guardrails_generated
    
    def train_on_task(
        self,
        task_id: int,
        dataset_path: str
    ) -> Dict:
        """
        Train on a single task from SeCA dataset.
        
        Args:
            task_id: Task ID to train on
            dataset_path: Path to SeCA dataset
        
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📚 TRAINING ON TASK {task_id}")
        logger.info(f"{'='*70}\n")
        
        self.current_task = task_id
        
        # Load dataset for this task
        dataset = SeCADataset(dataset_path, task_id=task_id)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x  # Return raw samples
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Setup scheduler
        num_training_steps = len(dataloader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        self.model.train()
        task_metrics = {
            'loss': [],
            'conflicts': 0,
            'guardrails': 0,
            'samples': 0
        }
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Task {task_id} Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Extract texts from batch
                texts = [sample['sentence'] for sample in batch]
                
                # SG-CL: Check conflicts and generate guard-rails
                augmented_texts, conflicts, guardrails = \
                    self._check_conflicts_and_generate_guardrails(texts)
                
                task_metrics['conflicts'] += conflicts
                task_metrics['guardrails'] += guardrails
                task_metrics['samples'] += len(texts)
                
                # Tokenize augmented batch
                inputs = self.tokenizer(
                    augmented_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['input_ids']
                )
                
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1
                
                # Logging
                epoch_loss += loss.item()
                task_metrics['loss'].append(loss.item())
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'conflicts': conflicts,
                    'guardrails': guardrails
                })
                
                # Periodic logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Conflicts: {task_metrics['conflicts']} | "
                        f"Guardrails: {task_metrics['guardrails']}"
                    )
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint for this task
        self._save_checkpoint(task_id, task_metrics)
        
        return task_metrics
    
    def _save_checkpoint(self, task_id: int, metrics: Dict):
        """Save model checkpoint and metrics."""
        output_dir = Path(self.config.output_dir) / f"task_{task_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save metrics
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✓ Checkpoint saved to {output_dir}")
    
    def train_sequential_tasks(
        self,
        dataset_path: str,
        num_tasks: int = 5
    ):
        """
        Train on multiple tasks sequentially (continual learning).
        
        Args:
            dataset_path: Path to SeCA dataset
            num_tasks: Number of tasks to train on
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 STARTING SG-CL SEQUENTIAL TRAINING")
        logger.info(f"{'='*70}\n")
        
        all_metrics = []
        
        for task_id in range(num_tasks):
            task_metrics = self.train_on_task(task_id, dataset_path)
            all_metrics.append(task_metrics)
            
            logger.info(f"\n✓ Task {task_id} complete")
            logger.info(f"  Conflicts detected: {task_metrics['conflicts']}")
            logger.info(f"  Guardrails generated: {task_metrics['guardrails']}")
            logger.info(f"  Samples processed: {task_metrics['samples']}\n")
        
        # Save final summary
        summary_path = Path(self.config.output_dir) / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'num_tasks': num_tasks,
                'total_steps': self.global_step,
                'task_metrics': all_metrics
            }, f, indent=2)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ SG-CL TRAINING COMPLETE")
        logger.info(f"{'='*70}\n")


def main():
    """Main training script for Kaggle."""
    # Configuration
    config = TrainingConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        batch_size=4,  # Adjust based on GPU memory
        num_epochs=3,
        enable_sid=True,
        enable_guardrails=True,
        output_dir="./sgcl_checkpoints"
    )
    
    # Initialize trainer
    trainer = SGCLTrainer(config)
    
    # Train on first 5 tasks
    trainer.train_sequential_tasks(
        dataset_path="sid/seca_10k_final.json",
        num_tasks=5
    )


if __name__ == "__main__":
    main()

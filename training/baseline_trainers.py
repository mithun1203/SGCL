"""
Baseline Training Implementations
==================================

Implements baseline methods for comparison:
1. Naive Fine-Tuning (no continual learning)
2. EWC (Elastic Weight Consolidation)
3. Replay Buffer

All use the same training skeleton as SG-CL but without SID/guard-rails.
"""

import torch
from torch.utils.data import DataLoader
import copy
from typing import Dict, List, Optional
from pathlib import Path
import json
import logging
from tqdm.auto import tqdm

from sgcl_trainer import TrainingConfig, SeCADataset, SGCLTrainer

logger = logging.getLogger(__name__)


class NaiveFinetuning(SGCLTrainer):
    """
    Baseline 1: Naive Fine-Tuning.
    
    Simply fine-tunes on each task sequentially without any
    continual learning mechanism.
    """
    
    def __init__(self, config: TrainingConfig):
        # Disable SID and guard-rails for baseline
        config.enable_sid = False
        config.enable_guardrails = False
        super().__init__(config)
        logger.info("📝 Naive Fine-Tuning Baseline initialized")


class EWCTrainer(SGCLTrainer):
    """
    Baseline 2: Elastic Weight Consolidation (EWC).
    
    Adds regularization term to preserve important weights from previous tasks.
    """
    
    def __init__(self, config: TrainingConfig, ewc_lambda: float = 5000):
        config.enable_sid = False
        config.enable_guardrails = False
        super().__init__(config)
        
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = {}
        self.optpar_dict = {}
        
        logger.info(f"⚖️  EWC Baseline initialized (λ={ewc_lambda})")
    
    def compute_fisher_information(self, dataloader):
        """Compute Fisher Information Matrix for current task."""
        logger.info("Computing Fisher Information Matrix...")
        
        self.model.train()
        fisher = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)
        
        for batch in tqdm(dataloader, desc="Computing Fisher"):
            texts = [sample['sentence'] for sample in batch]
            
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['input_ids']
            )
            
            loss = outputs.loss
            self.model.zero_grad()
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize Fisher information
        for name in fisher:
            fisher[name] /= len(dataloader)
        
        return fisher
    
    def ewc_loss(self):
        """Compute EWC regularization loss."""
        if not self.fisher_dict:
            return 0
        
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                loss += (self.fisher_dict[name] * 
                        (param - self.optpar_dict[name]) ** 2).sum()
        
        return self.ewc_lambda * loss
    
    def train_on_task(self, task_id: int, dataset_path: str) -> Dict:
        """Train on task with EWC regularization."""
        logger.info(f"\n{'='*70}")
        logger.info(f"📚 EWC TRAINING ON TASK {task_id}")
        logger.info(f"{'='*70}\n")
        
        dataset = SeCADataset(dataset_path, task_id=task_id)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Training loop
        self.model.train()
        task_metrics = {'loss': [], 'samples': 0}
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Task {task_id} Epoch {epoch+1}")
            
            for batch in progress_bar:
                texts = [sample['sentence'] for sample in batch]
                task_metrics['samples'] += len(texts)
                
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['input_ids']
                )
                
                # Standard loss + EWC regularization
                loss = outputs.loss + self.ewc_loss()
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                task_metrics['loss'].append(loss.item())
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # After training on task, update Fisher information
        fisher = self.compute_fisher_information(dataloader)
        
        # Update Fisher dict and optimal parameters
        if task_id == 0:
            self.fisher_dict = fisher
            self.optpar_dict = {name: param.data.clone() 
                               for name, param in self.model.named_parameters()}
        else:
            for name in fisher:
                self.fisher_dict[name] += fisher[name]
                self.optpar_dict[name] = self.model.state_dict()[name].clone()
        
        self._save_checkpoint(task_id, task_metrics)
        return task_metrics


class ReplayBufferTrainer(SGCLTrainer):
    """
    Baseline 3: Replay Buffer.
    
    Maintains a buffer of samples from previous tasks and
    replays them during training on new tasks.
    """
    
    def __init__(self, config: TrainingConfig, buffer_size: int = 500):
        config.enable_sid = False
        config.enable_guardrails = False
        super().__init__(config)
        
        self.buffer_size = buffer_size
        self.replay_buffer = []
        
        logger.info(f"🔄 Replay Buffer Baseline initialized (buffer_size={buffer_size})")
    
    def update_buffer(self, new_samples: List[Dict]):
        """Update replay buffer with new samples."""
        import random
        
        # Add new samples
        self.replay_buffer.extend(new_samples)
        
        # Maintain buffer size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = random.sample(self.replay_buffer, self.buffer_size)
        
        logger.info(f"Replay buffer size: {len(self.replay_buffer)}")
    
    def train_on_task(self, task_id: int, dataset_path: str) -> Dict:
        """Train on task with replay buffer."""
        logger.info(f"\n{'='*70}")
        logger.info(f"📚 REPLAY BUFFER TRAINING ON TASK {task_id}")
        logger.info(f"{'='*70}\n")
        
        dataset = SeCADataset(dataset_path, task_id=task_id)
        
        # Update replay buffer with samples from this task
        if len(dataset) > 0:
            import random
            sample_indices = random.sample(range(len(dataset)), 
                                          min(100, len(dataset)))
            new_samples = [dataset[i] for i in sample_indices]
            self.update_buffer(new_samples)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        self.model.train()
        task_metrics = {'loss': [], 'samples': 0, 'replay_samples': 0}
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Task {task_id} Epoch {epoch+1}")
            
            for batch in progress_bar:
                texts = [sample['sentence'] for sample in batch]
                
                # Add replay samples if buffer is not empty
                if self.replay_buffer and task_id > 0:
                    import random
                    replay_count = min(len(self.replay_buffer), 
                                      self.config.batch_size // 2)
                    replay_samples = random.sample(self.replay_buffer, replay_count)
                    texts.extend([s['sentence'] for s in replay_samples])
                    task_metrics['replay_samples'] += replay_count
                
                task_metrics['samples'] += len(texts)
                
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['input_ids']
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                task_metrics['loss'].append(loss.item())
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        self._save_checkpoint(task_id, task_metrics)
        return task_metrics


def main():
    """Train all baselines."""
    config = TrainingConfig(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        batch_size=4,
        num_epochs=3,
        output_dir="./baseline_checkpoints"
    )
    
    dataset_path = "sid/seca_10k_final.json"
    num_tasks = 5
    
    # Baseline 1: Naive Fine-Tuning
    logger.info("\n" + "="*70)
    logger.info("BASELINE 1: NAIVE FINE-TUNING")
    logger.info("="*70)
    naive_config = copy.deepcopy(config)
    naive_config.output_dir = "./baseline_naive"
    naive_trainer = NaiveFinetuning(naive_config)
    naive_trainer.train_sequential_tasks(dataset_path, num_tasks)
    
    # Baseline 2: EWC
    logger.info("\n" + "="*70)
    logger.info("BASELINE 2: EWC")
    logger.info("="*70)
    ewc_config = copy.deepcopy(config)
    ewc_config.output_dir = "./baseline_ewc"
    ewc_trainer = EWCTrainer(ewc_config, ewc_lambda=5000)
    ewc_trainer.train_sequential_tasks(dataset_path, num_tasks)
    
    # Baseline 3: Replay Buffer
    logger.info("\n" + "="*70)
    logger.info("BASELINE 3: REPLAY BUFFER")
    logger.info("="*70)
    replay_config = copy.deepcopy(config)
    replay_config.output_dir = "./baseline_replay"
    replay_trainer = ReplayBufferTrainer(replay_config, buffer_size=500)
    replay_trainer.train_sequential_tasks(dataset_path, num_tasks)


if __name__ == "__main__":
    main()

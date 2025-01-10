"""Training loop implementation for distributed sentiment analysis.

This module handles the training loop, metric calculation, and logging for
distributed training of the sentiment analysis model.
"""

import gc
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def setup_training_dirs() -> str:
    """Create necessary directories and return timestamp string."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for dir_name in ['logs', 'results']:
        Path(dir_name).mkdir(exist_ok=True)
    return timestamp

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    global_rank: int,
    gradient_accumulation_steps: int = 4
) -> Dict[str, Any]:
    """Train the model using distributed training."""
    timestamp = setup_training_dirs()
    
    logger.info(f"Starting training on rank {global_rank}")
    logger.info(f"Sampler config: shuffle={train_loader.sampler.shuffle}, "
                f"replicas={train_loader.sampler.num_replicas}, "
                f"rank={train_loader.sampler.rank}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    results = {
        'rank': global_rank,
        'timestamp': timestamp,
        'epochs': []
    }
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Log batch indices periodically
            if batch_idx % 5 == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                # Get first token sequences for display (first 5 tokens)
                sample_tokens = batch["input_ids"][0][:5].tolist()
                sample_label = batch["label"][0].item()
                sample_indices = list(train_loader.sampler)[
                    batch_idx * train_loader.batch_size:
                    (batch_idx + 1) * train_loader.batch_size
                ]
                logger.info(
                    f"Rank {global_rank} - Progress: {progress:.1f}% "
                    f"[{batch_idx + 1}/{len(train_loader)}] "
                    f"Sample: tokens={sample_tokens}, label={sample_label} "
                    f"Batch indices: {sample_indices}"
                )
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels) / gradient_accumulation_steps
            
            loss.backward()
            
            train_loss += loss.item() * gradient_accumulation_steps
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            del outputs, loss, input_ids, attention_mask, labels
            gc.collect()

        # Calculate per-machine metrics
        avg_rank_train_loss = train_loss / train_total
        rank_train_accuracy = 100. * train_correct / train_total if train_total > 0 else 0
        
        # Sync tensors for combined metrics
        total_loss_tensor = torch.tensor([train_loss]).to(device)
        total_examples_tensor = torch.tensor([train_total]).to(device)
        total_correct_tensor = torch.tensor([train_correct]).to(device)
        
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        
        combined_train_loss = total_loss_tensor.item() / total_examples_tensor.item()
        combined_train_accuracy = 100. * total_correct_tensor.item() / total_examples_tensor.item()

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                del outputs, loss, input_ids, attention_mask, labels
        
        # Calculate per-machine validation metrics
        avg_rank_val_loss = val_loss / val_total
        rank_val_accuracy = 100. * val_correct / val_total if val_total > 0 else 0
        
        # Sync validation metrics
        total_val_loss_tensor = torch.tensor([val_loss]).to(device)
        total_val_examples_tensor = torch.tensor([val_total]).to(device)
        total_val_correct_tensor = torch.tensor([val_correct]).to(device)
        
        dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_val_examples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_val_correct_tensor, op=dist.ReduceOp.SUM)
        
        combined_val_loss = total_val_loss_tensor.item() / total_val_examples_tensor.item()
        combined_val_accuracy = 100. * total_val_correct_tensor.item() / total_val_examples_tensor.item()

        # Log both per-machine and combined metrics
        logger.info(
            f"Rank {global_rank} Metrics - Epoch {epoch+1}:\n"
            f"Training: Loss={avg_rank_train_loss:.4f}, Acc={rank_train_accuracy:.2f}%\n"
            f"Validation: Loss={avg_rank_val_loss:.4f}, Acc={rank_val_accuracy:.2f}%"
        )
        logger.info(
            f"Combined Metrics - Epoch {epoch+1}:\n"
            f"Training: Loss={combined_train_loss:.4f}, Acc={combined_train_accuracy:.2f}%\n"
            f"Validation: Loss={combined_val_loss:.4f}, Acc={combined_val_accuracy:.2f}%"
        )

        # Store epoch results
        results['epochs'].append({
            'epoch': epoch + 1,
            'rank_train_loss': avg_rank_train_loss,
            'rank_train_accuracy': rank_train_accuracy,
            'rank_val_loss': avg_rank_val_loss,
            'rank_val_accuracy': rank_val_accuracy,
            'combined_train_loss': combined_train_loss,
            'combined_train_accuracy': combined_train_accuracy,
            'combined_val_loss': combined_val_loss,
            'combined_val_accuracy': combined_val_accuracy
        })
        
        # Save results from all ranks
        results_path = Path('results') / f'results_rank{global_rank}_{timestamp}.json'
        with results_path.open('w') as f:
            json.dump(results, f, indent=4)
    
    logger.info(f"Training completed on rank {global_rank}")
    return results
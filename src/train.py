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

class TrainingMetrics:
    """Helper class to track and synchronize training metrics across processes."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.loss = 0.0
        self.correct = 0
        self.total = 0
    
    def update(self, loss: float, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """Update metrics with batch results."""
        self.loss += loss
        self.total += labels.size(0)
        self.correct += predictions.eq(labels).sum().item()
    
    def synchronize(self) -> Dict[str, float]:
        """Synchronize metrics across all processes.
        
        Returns:
            Dictionary containing synchronized loss and accuracy metrics
        """
        # Prepare tensors for synchronization
        metrics_tensor = torch.tensor(
            [self.loss, self.total, self.correct],
            dtype=torch.float64,
            device=self.device
        )
        
        # Synchronize across processes
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        # Calculate combined metrics
        total_loss, total_examples, total_correct = metrics_tensor.tolist()
        
        return {
            'loss': total_loss / total_examples if total_examples > 0 else 0.0,
            'accuracy': 100.0 * total_correct / total_examples if total_examples > 0 else 0.0
        }

def setup_training_dirs() -> str:
    """Create necessary directories and return timestamp string.
    
    Returns:
        Timestamp string for unique file naming
    """
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
    gradient_accumulation_steps: int = 4
) -> Dict[str, Any]:
    """Train the model using distributed training.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        num_epochs: Number of epochs to train
        gradient_accumulation_steps: Number of steps to accumulate gradients
        
    Returns:
        Dictionary containing training results and metrics
    """
    # Setup
    rank = dist.get_rank()
    timestamp = setup_training_dirs()
    
    logger.info(f"Starting training on rank {rank}")
    logger.info(f"Sampler config: shuffle={train_loader.sampler.shuffle}, "
                f"replicas={train_loader.sampler.num_replicas}, "
                f"rank={train_loader.sampler.rank}")
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    train_metrics = TrainingMetrics(device)
    val_metrics = TrainingMetrics(device)
    
    # Store results
    results = {
        'rank': rank,
        'timestamp': timestamp,
        'epochs': []
    }
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_metrics.reset()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Log progress periodically
            if batch_idx % 5 == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                logger.info(
                    f"Progress: {progress:.1f}% [{batch_idx + 1}/{len(train_loader)}] "
                    f"Sample batch size: {len(batch['input_ids'])}"
                )
            
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels) / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            with torch.no_grad():
                predictions = outputs.max(1)[1]
                train_metrics.update(loss.item() * gradient_accumulation_steps,
                                   predictions, labels)
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Clear memory
            del outputs, loss, input_ids, attention_mask, labels
            gc.collect()

        # Validation phase
        model.eval()
        val_metrics.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                predictions = outputs.max(1)[1]
                val_metrics.update(loss.item(), predictions, labels)
                
                del outputs, loss, input_ids, attention_mask, labels
        
        # Synchronize and log metrics
        train_results = train_metrics.synchronize()
        val_results = val_metrics.synchronize()
        
        logger.info(
            f"Epoch {epoch+1} Results:\n"
            f"Training - Loss: {train_results['loss']:.4f}, "
            f"Accuracy: {train_results['accuracy']:.2f}%\n"
            f"Validation - Loss: {val_results['loss']:.4f}, "
            f"Accuracy: {val_results['accuracy']:.2f}%"
        )
        
        # Store epoch results
        results['epochs'].append({
            'epoch': epoch + 1,
            'train_loss': train_results['loss'],
            'train_accuracy': train_results['accuracy'],
            'val_loss': val_results['loss'],
            'val_accuracy': val_results['accuracy']
        })
        
        # Save results from rank 0
        if rank == 0:
            results_path = Path('results') / f'results_{timestamp}.json'
            with results_path.open('w') as f:
                json.dump(results, f, indent=4)
    
    logger.info(f"Training completed on rank {rank}")
    return results
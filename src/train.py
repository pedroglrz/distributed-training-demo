import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import torch.distributed as dist
import gc
import json
from datetime import datetime
import logging

def setup_logging(rank):
    """Setup basic logging for each process"""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/process_rank{rank}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - Rank %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return timestamp

def train_model(model, train_loader, val_loader, device, num_epochs, gradient_accumulation_steps=4):
    # Setup process-specific logging
    rank = dist.get_rank()
    timestamp = setup_logging(rank)
    
    # Log sampler configuration once at start
    logging.info(f"{rank} - Sampler config: shuffle={train_loader.sampler.shuffle}, replicas={train_loader.sampler.num_replicas}, rank={train_loader.sampler.rank}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Store results
    results = {
        'rank': rank,
        'timestamp': timestamp,
        'epochs': []
    }
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        logging.info(f"{rank} - Starting epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()
        
        # Progress tracking
        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            # Log sample of processed data every 5 batches
            if batch_idx % 5 == 0:
                progress = (batch_idx + 1) / total_batches * 100
                # Get first token sequences for display (first 5 tokens)
                sample_tokens = batch["input_ids"][0][:5].tolist()
                sample_label = batch["label"][0].item()
                logging.info(
                    f"{rank} - Progress: {progress:.1f}% [{batch_idx + 1}/{total_batches}] "
                    f"Sample: tokens={sample_tokens}, label={sample_label}"
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

        # Calculate rank-specific metrics
        avg_rank_train_loss = train_loss / train_total
        rank_train_accuracy = 100. * train_correct / train_total if train_total > 0 else 0
        
        # Prepare tensors for synchronization
        total_loss_tensor = torch.tensor([train_loss]).to(device)
        total_examples_tensor = torch.tensor([train_total]).to(device)
        total_correct_tensor = torch.tensor([train_correct]).to(device)
        
        # Synchronize metrics across processes
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        
        # Calculate combined metrics
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
        
        # Calculate rank-specific validation metrics
        avg_rank_val_loss = val_loss / val_total
        rank_val_accuracy = 100. * val_correct / val_total if val_total > 0 else 0
        
        # Prepare tensors for synchronization
        total_val_loss_tensor = torch.tensor([val_loss]).to(device)
        total_val_examples_tensor = torch.tensor([val_total]).to(device)
        total_val_correct_tensor = torch.tensor([val_correct]).to(device)
        
        # Synchronize validation metrics
        dist.all_reduce(total_val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_val_examples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_val_correct_tensor, op=dist.ReduceOp.SUM)
        
        # Calculate combined validation metrics
        combined_val_loss = total_val_loss_tensor.item() / total_val_examples_tensor.item()
        combined_val_accuracy = 100. * total_val_correct_tensor.item() / total_val_examples_tensor.item()

        # Simplified logging - one line per metric group
        logging.info(
            f"{rank} - Epoch {epoch+1} Training: "
            f"Loss={avg_rank_train_loss:.4f}, Acc={rank_train_accuracy:.2f}% | "
            f"Val: Loss={avg_rank_val_loss:.4f}, Acc={rank_val_accuracy:.2f}%"
        )
        logging.info(
            f"Combined - Epoch {epoch+1} Training: "
            f"Loss={combined_train_loss:.4f}, Acc={combined_train_accuracy:.2f}% | "
            f"Val: Loss={combined_val_loss:.4f}, Acc={combined_val_accuracy:.2f}%"
        )

        # Store epoch results
        epoch_results = {
            'epoch': epoch + 1,
            'combined_train_loss': combined_train_loss,
            'combined_train_accuracy': combined_train_accuracy,
            'combined_val_loss': combined_val_loss,
            'combined_val_accuracy': combined_val_accuracy
        }
        results['epochs'].append(epoch_results)
        
        # Save results from rank 0 only
        if rank == 0:
            with open(f'results/results_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=4)
    
    logging.info(f"{rank} - Training completed")
    return results
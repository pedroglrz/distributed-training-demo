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
        
        # Log start of epoch for each process
        logging.info(f"{rank} - Starting epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()
        
        # Custom progress tracking
        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            # Log progress every few batches
            if batch_idx % 5 == 0:
                progress = (batch_idx + 1) / total_batches * 100
                logging.info(f"{rank} - Progress: {progress:.1f}% [{batch_idx + 1}/{total_batches}]")
                logging.info(f"Node {rank} - Processing batch {batch_idx}")
            
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
        avg_rank_train_loss = train_loss / train_total  # Average loss per example for this rank
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
        combined_train_loss = total_loss_tensor.item() / total_examples_tensor.item()  # Average loss per example across all ranks
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
        avg_rank_val_loss = val_loss / val_total  # Average loss per example for this rank
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
        combined_val_loss = total_val_loss_tensor.item() / total_val_examples_tensor.item()  # Average loss per example across all ranks
        combined_val_accuracy = 100. * total_val_correct_tensor.item() / total_val_examples_tensor.item()

        #logging
        logging.info(f"{rank} - Epoch {epoch+1} Avg Training Loss per example: {avg_rank_train_loss:.4f}, Accuracy: {rank_train_accuracy:.2f}%")
        logging.info(f"{rank} - Epoch {epoch+1} Avg Validation Loss per example: {avg_rank_val_loss:.4f}, Accuracy: {rank_val_accuracy:.2f}%")
        logging.info(f"Combined - Epoch {epoch+1} Avg Training Loss per example: {combined_train_loss:.4f}, Accuracy: {combined_train_accuracy:.2f}%")
        logging.info(f"Combined - Epoch {epoch+1} Validation Loss per example: {combined_val_loss:.4f}, Accuracy: {combined_val_accuracy:.2f}%")

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
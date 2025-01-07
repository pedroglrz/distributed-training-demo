import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import torch.distributed as dist
import gc

def train_model(model, train_loader, val_loader, device, num_epochs, gradient_accumulation_steps=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"[Process {os.getpid()}] Training")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels) / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            train_loss += loss.item() * gradient_accumulation_steps
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Clear memory
            del outputs, loss, input_ids, attention_mask, labels
            if batch_idx % 2 == 0:  # Every 2 batches
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Final step for remaining gradients
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        train_accuracy = 100. * train_correct / train_total if train_total > 0 else 0
        
        # Synchronize metrics
        train_loss_tensor = torch.tensor([train_loss]).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / dist.get_world_size()
        
        if dist.get_rank() == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Training Accuracy: {train_accuracy:.2f}%")
        
        # Validation phase with memory optimization
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Clear memory
                del outputs, loss, input_ids, attention_mask, labels
            
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Synchronize validation metrics
        val_metrics = torch.tensor([val_loss, val_correct, val_total], dtype=torch.float32).to(device)
        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        val_loss, val_correct, val_total = val_metrics.tolist()
        
        val_loss = val_loss / dist.get_world_size()
        
        if val_total > 0:
            val_accuracy = 100. * val_correct / val_total
            if dist.get_rank() == 0:
                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Validation Accuracy: {val_accuracy:.2f}%\n")
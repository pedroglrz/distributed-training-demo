import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import torch.distributed as dist

def train_model(model, train_loader, val_loader, device, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        
    for epoch in range(num_epochs):
        # Important: set epoch for sampler
        train_loader.sampler.set_epoch(epoch)        
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"[Process {os.getpid()}] Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_accuracy = 100. * train_correct / train_total

        # Add after calculating train_loss, train_correct, etc:
        train_loss_tensor = torch.tensor([train_loss]).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / dist.get_world_size()

        # Only print on main process
        if dist.get_rank() == 0:
            print(f"Training Loss: {train_loss:.4f}")                
            print(f"Training Accuracy: {train_accuracy:.2f}%")

        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)  # Fixed from outputs to logits
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Reduce validation metrics
        val_loss_tensor = torch.tensor([val_loss]).to(device)
        val_correct_tensor = torch.tensor([val_correct]).to(device)
        val_total_tensor = torch.tensor([val_total]).to(device)

        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)

        val_loss = val_loss_tensor.item() / dist.get_world_size()
        val_correct = val_correct_tensor.item()
        val_total = val_total_tensor.item()

        if val_total > 0:
            val_accuracy = 100. * val_correct / val_total
            if dist.get_rank() == 0:  # Only print on main process
                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Validation Accuracy: {val_accuracy:.2f}%")
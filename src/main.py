import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from model import TransformerClassifier
from train import train_model
from imdb_dataset import IMDBDataset


def setup(rank, world_size):
    """
    Initialize the distributed environment for single machine.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Using gloo backend for CPU training
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    torch.set_num_threads(1)  # Important for CPU training

def cleanup():
    dist.destroy_process_group()

def train_process(rank, world_size, model_name, max_length, batch_size, num_epochs):
    # Initialize process group
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"Training on device: {device}")
    
    # Create datasets
    train_dataset = IMDBDataset(
        split="train",
        max_length=max_length,
        subset_size=256,
        model_name=model_name,
    )
    val_dataset = IMDBDataset(
        split="test",
        max_length=max_length,
        subset_size=32,
        model_name=model_name,
    )

    # Create samplers with deterministic shuffling
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,  # Increased for better CPU utilization
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True        
    )

    # Create model
    model = TransformerClassifier(model_name=model_name).to(device)
    
    # Properly wrap model in DDP
    model = DistributedDataParallel(
        model,
        device_ids=None,  # None for CPU
        output_device=None  # None for CPU
    )

    # Train model
    train_model(model, train_loader, val_loader, device, num_epochs)
    
    # Cleanup
    cleanup()

def main():
    model_name = "distilbert-base-uncased"
    max_length = 256
    batch_size = 4  # Reduced for CPU training
    num_epochs = 2
    world_size = 2  # Number of processes

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Launch processes
    mp.spawn(
        train_process,
        args=(world_size, model_name, max_length, batch_size, num_epochs),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
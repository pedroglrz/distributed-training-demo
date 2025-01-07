import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import TransformerClassifier
from train import train_model
from imdb_dataset import IMDBDataset

def setup():
    """
    Initialize the distributed environment.
    """
    dist.init_process_group("gloo")
    torch.set_num_threads(1)

def cleanup():
    dist.destroy_process_group()

def main():
    # The launcher sets LOCAL_RANK and WORLD_SIZE environment variables
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Setup distributed
    setup()
    
    # Model parameters
    model_name = "distilbert-base-uncased"
    max_length = 256
    batch_size = 4
    num_epochs = 2
    
    # Set device
    device = torch.device('cpu')
    
    if local_rank == 0:
        print(f"Training on device: {device}")
        print(f"World size: {world_size}")
    
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

    # Create samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=42
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
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
    
    # Wrap model in DDP
    model = DistributedDataParallel(
        model,
        device_ids=None,
        output_device=None
    )

    # Train model
    train_model(model, train_loader, val_loader, device, num_epochs)
    
    # Cleanup
    cleanup()

if __name__ == "__main__":
    main()
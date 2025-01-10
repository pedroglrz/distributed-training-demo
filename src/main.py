import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import TransformerClassifier
from train import train_model
from imdb_dataset import IMDBDataset
from datetime import datetime

def setup():
    """
    Initialize the distributed environment.
    """
    # Ensure all required environment variables are set
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', '2')
    os.environ['RANK'] = os.environ.get('RANK', '0')
    
    # Initialize the process group
    dist.init_process_group(
        "gloo",
        timeout=datetime.timedelta(minutes=30)
    )
    
    # Set deterministic behavior
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def cleanup():
    dist.destroy_process_group()

def main():
    # Get environment variables with defaults
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))  # Add default value
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Setup distributed
    setup()
    
    # Add detailed logging
    print(f"Process started with:")
    print(f"- Local Rank: {local_rank}")
    print(f"- Node Rank: {node_rank}")
    print(f"- World Size: {world_size}")
    print(f"- Master addr: {os.environ.get('MASTER_ADDR')}")
    print(f"- Master port: {os.environ.get('MASTER_PORT')}")
    
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

    #log
    print(f"Total dataset size: {len(train_dataset)}")
    print(f"Samples per process: {len(train_loader)}")

    # Print first few indices that this process will handle
    indices = list(train_loader.sampler)[:5]
    print(f"First 5 indices for process {local_rank}: {indices}")    

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
"""Distributed training entry point for IMDB sentiment analysis.

This module handles the distributed training setup and coordination between processes,
implementing DistributedDataParallel (DDP) for efficient multi-node training.
"""

import datetime
import logging
import os
from typing import Tuple, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import TransformerClassifier
from train import train_model
from imdb_dataset import IMDBDataset

logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """Configure logging for the distributed training process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def setup_distributed() -> None:
    """Initialize the distributed training environment.
    
    Ensures all required environment variables are set and initializes the
    process group for distributed training.
    """
    # Set default environment variables if not present
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

def cleanup_distributed() -> None:
    """Clean up the distributed training environment."""
    dist.destroy_process_group()

def get_rank_info() -> Tuple[int, int, int, int]:
    """Get process rank information for distributed training.
    
    Returns:
        Tuple containing local_rank, node_rank, global_rank, and world_size
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Calculate global rank
    procs_per_node = int(os.environ.get("NPROC_PER_NODE", 1))
    global_rank = node_rank * procs_per_node + local_rank
    
    return local_rank, node_rank, global_rank, world_size

def create_dataloaders(
    global_rank: int,
    world_size: int,
    batch_size: int,
    max_length: int,
    model_name: str
) -> Tuple[DataLoader, DataLoader]:
    """Create distributed DataLoaders for training and validation.
    
    Args:
        global_rank: Global rank of current process
        world_size: Total number of processes
        batch_size: Batch size per process
        max_length: Maximum sequence length
        model_name: Name of pretrained model for tokenization
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
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

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=42
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False
    )

    # Create and return dataloaders
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
    
    logger.info(f"Dataset sizes - Total: {len(train_dataset)}, "
                f"Samples per process: {len(train_loader)}")
    
    return train_loader, val_loader

def main() -> None:
    """Main entry point for distributed training."""
    # Setup logging
    setup_logging()
    
    # Initialize distributed environment
    setup_distributed()
    
    # Get rank information
    local_rank, node_rank, global_rank, world_size = get_rank_info()
    
    # Log distributed setup
    logger.info(
        f"Distributed setup:\n"
        f"- Local Rank: {local_rank}\n"
        f"- Node Rank: {node_rank}\n"
        f"- Global Rank: {global_rank}\n"
        f"- World Size: {world_size}\n"
        f"- Master addr: {os.environ.get('MASTER_ADDR')}\n"
        f"- Master port: {os.environ.get('MASTER_PORT')}"
    )
    
    # Training parameters
    model_name = "distilbert-base-uncased"
    max_length = 256
    batch_size = 4
    num_epochs = 2
    
    # Set device
    device = torch.device('cpu')
    if global_rank == 0:
        logger.info(f"Training on device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        global_rank, world_size, batch_size, max_length, model_name
    )
    
    # Create and wrap model in DDP
    model = TransformerClassifier(model_name=model_name).to(device)
    model = DistributedDataParallel(
        model,
        device_ids=None,
        output_device=None
    )

    # Train model
    try:
        train_model(model, train_loader, val_loader, device, num_epochs)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
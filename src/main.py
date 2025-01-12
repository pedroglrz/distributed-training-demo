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
    """Initialize the distributed training environment."""
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['WORLD_SIZE'] = os.environ.get('WORLD_SIZE', '2')
    os.environ['RANK'] = os.environ.get('RANK', '0')
    
    dist.init_process_group(
        "gloo",
        timeout=datetime.timedelta(minutes=30)
    )
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def cleanup_distributed() -> None:
    """Clean up the distributed training environment."""
    dist.destroy_process_group()

def get_rank_info() -> Tuple[int, int, int, int]:
    """Get process rank information for distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
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
    """Create distributed DataLoaders for training and validation."""
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
    
    logger.info(f"Total dataset size: {len(train_dataset)}")
    logger.info(f"Samples per process: {len(train_loader)}")

    # Print first few indices that this process will handle
    indices = list(train_loader.sampler)[:5]
    logger.info(f"First 5 indices for process {global_rank}: {indices}")
    
    return train_loader, val_loader

def main() -> None:
    """Main entry point for distributed training."""
    setup_logging()
    setup_distributed()
    
    local_rank, node_rank, global_rank, world_size = get_rank_info()
    
    logger.info(
        f"Distributed setup:\n"
        f"- Local Rank: {local_rank}\n"
        f"- Node Rank: {node_rank}\n"
        f"- Global Rank: {global_rank}\n"
        f"- World Size: {world_size}\n"
        f"- Master addr: {os.environ.get('MASTER_ADDR')}\n"
        f"- Master port: {os.environ.get('MASTER_PORT')}"
    )
    
    model_name = "distilbert-base-uncased"
    max_length = 256
    batch_size = 4
    num_epochs = 2
    
    device = torch.device('cpu')
    if global_rank == 0:
        logger.info(f"Training on device: {device}")
    
    train_loader, val_loader = create_dataloaders(
        global_rank, world_size, batch_size, max_length, model_name
    )
    
    model = TransformerClassifier(model_name=model_name).to(device)
    model = DistributedDataParallel(
        model,
        device_ids=None,
        output_device=None
    )

    try:
        train_model(model, train_loader, val_loader, device, num_epochs, global_rank)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
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
from logging_utils import ProcessLogger

logger = ProcessLogger()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_process(rank, world_size, model_name, max_length, batch_size, num_epochs,logger):
    setup(rank, world_size)
    
    # Create datasets
    logger.log(f"[Process {os.getpid()}] Creating datasets...")
    train_dataset = IMDBDataset(
        split="train",
        max_length=max_length,
        subset_size=100,  # Adjust as needed
        model_name=model_name,
        logger = logger
    )
    val_dataset = IMDBDataset(
        split="test",
        max_length=max_length,
        subset_size=10,  # Adjust as needed
        model_name=model_name,
        logger=logger,
    )

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        drop_last=True        
    )

    # Create model
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = TransformerClassifier(model_name=model_name).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])

    # Train model
    train_model(model, train_loader, val_loader, device, num_epochs, logger)
    cleanup()

def main():
    # Configuration
    model_name = "bert-base-uncased"
    max_length = 512
    batch_size = 16
    num_epochs = 2
    world_size = 3  # Number of processes

    mp.spawn(
        train_process,
        args=(world_size, model_name, max_length, batch_size, num_epochs,logger),
        nprocs=world_size
    )

if __name__ == "__main__":
    main()
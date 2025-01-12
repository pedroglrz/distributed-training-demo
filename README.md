# Distributed Training Demo with PyTorch DDP

This project demonstrates distributed training concepts using PyTorch's DistributedDataParallel (DDP) through a sentiment analysis task on the IMDB dataset. It specifically highlights:

- Data preprocessing redundancy across nodes
- Training on disjoint examples per batch
- Per-node and combined metrics tracking

## Setup

1. **Environment Setup**

```bash
# Create and activate virtual environment with make file
make setup-venv
source ~/envs/dlp/bin/activate
```

2. **AWS Configuration**

This demo is designed to run on two AWS instances. Configure your instances:
- Set up two EC2 instances with Python and PyTorch installed (I used 2xt2.medium's with 32GB of storage each)
- Ensure instances can communicate (same security group/network)
- Note the private IP of the master node

## Running the Training

1. **On Master Node (Node 0)**:

```bash
# Update MASTER_IP in launch.sh with your master node's private IP
./launch.sh 0
```

2. **On Worker Node (Node 1)**:

```bash
# Make sure MASTER_IP matches the master node's IP
./launch.sh 1
```

## Outputs

The training shows:
1. Node initialization and data loading details
2. Per-batch progress with sample indices and tokens
3. Per-node and combined metrics for each epoch

Example output (for node of rank 0):
```
2024-01-10 12:34:56 - [INFO] - Rank 0 - Progress: 20.0% [10/50] Sample: tokens=[101, 2054, 3082, 2024, 2003], label=1 Batch indices: [0, 4, 8, 12]
2024-01-10 12:34:56 - [INFO] - Rank 0 Metrics - Epoch 1:
Training: Loss=0.6943, Acc=52.34%
Validation: Loss=0.6898, Acc=54.21%
```
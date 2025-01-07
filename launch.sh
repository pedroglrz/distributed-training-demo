#!/bin/bash

# Configuration for AWS instances
MASTER_IP="172.31.21.7"  # Replace with dlp-node-0's private IP
NODE_RANK=$1          # Will be 0 for master node, 1 for worker
NUM_NODES=2
PROCS_PER_NODE=1
MASTER_PORT=12355

# Activate virtual environment if needed
# source venv/bin/activate

# Launch training
python -m torch.distributed.launch \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$PROCS_PER_NODE \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    src/main.py
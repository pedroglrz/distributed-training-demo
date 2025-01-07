#!/bin/bash

# Configuration for local testing
MASTER_IP="localhost"  # Using localhost for single-machine testing
NODE_RANK=$1          # Will be 0 for first process, 1 for second
NUM_NODES=2
PROCS_PER_NODE=1
MASTER_PORT=12355

# Launch training with correct path to main.py
python -m torch.distributed.launch \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$PROCS_PER_NODE \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    src/main.py
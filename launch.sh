#!/bin/bash

# Configuration for AWS instances
MASTER_IP="172.31.21.7"  # Replace with dlp-node-0's private IP
NODE_RANK=$1  # Will be 0 for master node, 1 for worker
NUM_NODES=2
PROCS_PER_NODE=1
MASTER_PORT=12355

# Add error checking
if [ -z "$MASTER_IP" ]; then
    echo "Error: MASTER_IP not set"
    exit 1
fi

if [ -z "$NODE_RANK" ]; then
    echo "Error: NODE_RANK not provided"
    exit 1
fi

# Log the configuration
echo "Starting distributed training with:"
echo "- Master IP: $MASTER_IP"
echo "- Node Rank: $NODE_RANK"
echo "- Num Nodes: $NUM_NODES"
echo "- Processes per Node: $PROCS_PER_NODE"
echo "- Master Port: $MASTER_PORT"

# Launch training with explicit environment variables
NCCL_DEBUG=INFO python -m torch.distributed.launch \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$PROCS_PER_NODE \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    src/main.py    
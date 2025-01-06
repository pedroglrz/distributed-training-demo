#!/bin/bash

# Set master IP at the top of the script
MASTER_IP="10.0.0.1"  # Replace with your actual master node IP
NODE_RANK=$1

python \
    --nnodes=3 \
    --node_rank=$NODE_RANK \
    --nproc_per_node=1 \
    --master_addr=$MASTER_IP \
    --master_port=12355 \
    main.py
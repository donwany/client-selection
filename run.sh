#!/bin/bash

# Head node
python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=0 \
--master_addr=head-node-public-ip \
--master_port=1234 \
examples/pytorch/distributed/resnet/main.py \
--backend=nccl --use_syn --batch_size=16 --arch=resnet152

# Worker node 0
python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=1 \
--master_addr=head-node-public-ip \
--master_port=1234 \
examples/pytorch/distributed/resnet/main.py \
--backend=nccl --use_syn --batch_size=16 --arch=resnet152

# Worker node 1
python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=3 --node_rank=2 \
--master_addr=head-node-public-ip \
--master_port=1234 \
examples/pytorch/distributed/resnet/main.py \
--backend=nccl --use_syn --batch_size=16 --arch=resnet152
#!/bin/bash

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# run fedavg with client selection
# h0,h1,.. are hosts names of the clusters
# replace them either by IP address or your own host names

#pdsh -R ssh -w h0,h1,h2 "pkill python3.5"
#
#pdsh -R ssh -w h0,h1,h2 "python3.5 dnn/train_dnn.py \
#                        --constantE --lr 0.005 --bs 64 --localE 30 \
#                        --alpha 2 --dataset fmnist --seltype rand \
#                        --powd 2 --ensize 100 --fracC 0.03 --size 3 \
#                        --save -p --optimizer fedavg --model MLP\
#                        --rank %n  --backend nccl --initmethod tcp://h0 \
#                        --rounds 300 --seed 2 --NIID --print_freq 50"
#
#pdsh -R ssh -w h0,h1,h2 "python3.5 dnn/train_dnn.py \
#                        --constantE --lr 0.005 --bs 64 --localE 30 \
#                        --alpha 2 --dataset fmnist --seltype pow-d \
#                        --powd 6 --ensize 100 --fracC 0.03 --size 3 \
#                        --save -p --optimizer fedavg --model MLP \
#                        --rank %n  --backend nccl --initmethod tcp://h0 \
#                        --rounds 300 --seed 2 --NIID --print_freq 50"


python3.8 \
      train_dnn.py \
      --constantE \
      --lr 0.005 \
      --bs 64 \
      --localE 30 \
      --alpha 2 \
      --dataset fmnist \
      --seltype rand \
      --powd 2 \
      --ensize 100 \
      --fracC 0.03 \
      --size 3 \
      --save -p \
      --optimizer fedavg \
      --model MLP \
      --rank 0  \
      --backend gloo \
      --initmethod 'file:///Users/tsiameh/Desktop/paper-writing/FLFSL/FL-Client-Selection' \
      --rounds 300 \
      --seed 2 \
      --out_fname '/Users/tsiameh/Desktop/paper-writing/FLFSL/FL-Client-Selection' \
      --NIID \
      --print_freq 50


#python -m torch.distributed.run \
#      --nproc_per_node 3 \
#      train_dnn.py \
#      --constantE \
#      --lr 0.005 \
#      --bs 64 \
#      --localE 30 \
#      --alpha 2 \
#      --dataset fmnist \
#      --seltype rand \
#      --powd 2 \
#      --ensize 100 \
#      --fracC 0.03 \
#      --size 3 \
#      --save -p \
#      --optimizer fedavg \
#      --model MLP \
#      --rank 0  \
#      --backend nccl \
#      --initmethod 'env://' \
#      --rounds 300 \
#      --seed 2 \
#      --out_fname '/content/fl-client-selection' \
#      --NIID \
#      --print_freq 50


#python -m torch.distributed.run \
#      --nproc_per_node 3 \
#      train_dnn.py \
#      --constantE \
#      --lr 0.005 \
#      --bs 64 \
#      --localE 30 \
#      --alpha 2 \
#      --dataset fmnist \
#      --seltype pow-d \
#      --powd 6 \
#      --ensize 100 \
#      --fracC 0.03 \
#      --size 3 \
#      --save -p \
#      --optimizer fedavg \
#      --model MLP \
#      --rank 1  \
#      --backend nccl \
#      --initmethod "env://" \
#      --rounds 300 \
#      --seed 2 \
#      --out_fname '/content/fl-client-selection' \
#      --NIID \
#      --print_freq 50
#
#python -m torch.distributed.run \
#      --nproc_per_node 3 \
#      train_dnn.py \
#      --constantE \
#      --lr 0.005 \
#      --bs 64 \
#      --localE 30 \
#      --alpha 2 \
#      --dataset fmnist \
#      --seltype pow-d \
#      --powd 9 \
#      --ensize 100 \
#      --fracC 0.03 \
#      --size 3 \
#      --save -p \
#      --optimizer fedavg \
#      --model MLP \
#      --rank 1  \
#      --backend nccl \
#      --initmethod "env://" \
#      --rounds 300 \
#      --seed 2 \
#      --out_fname '/content/fl-client-selection' \
#      --NIID \
#      --print_freq 50
#
#
#python -m torch.distributed.run \
#      --nproc_per_node 3 \
#      train_dnn.py \
#      --constantE \
#      --lr 0.005 \
#      --bs 64 \
#      --localE 30 \
#      --alpha 2 \
#      --dataset fmnist \
#      --seltype pow-d \
#      --powd 15 \
#      --ensize 100 \
#      --fracC 0.03 \
#      --size 3 \
#      --save -p \
#      --optimizer fedavg \
#      --model MLP \
#      --rank 1  \
#      --backend nccl \
#      --initmethod "env://" \
#      --rounds 300 \
#      --seed 2 \
#      --out_fname '/content/fl-client-selection' \
#      --NIID \
#      --print_freq 50

# when alpha = 0.3
#python -m torch.distributed.run \
#      --nproc_per_node 4 \
#      train_dnn.py \
#      --constantE \
#      --lr 0.005 \
#      --bs 64 \
#      --localE 30 \
#      --alpha 0.3 \
#      --dataset fmnist \
#      --seltype rand \
#      --powd 2 \
#      --ensize 100 \
#      --fracC 0.03 \
#      --size 4 \
#      --save -p \
#      --optimizer fedavg \
#      --model MLP \
#      --rank 1  \
#      --backend nccl \
#      --initmethod "env://" \
#      --rounds 300 \
#      --seed 2 \
#      --out_fname '/content/fl-client-selection' \
#      --NIID \
#      --print_freq 50

# support@ece.msstate.edu
# IP: 130.18.65.39
# open ece account for me



#!/bin/bash

TEMP=$1
RECONPATH=$2
PRETRAINED=$3

cd finetune
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    --nnodes=1 --nproc_per_node=4 train_adapter.py \
    --loss contrastive \
    --data-path VisDial \
    --temp ${TEMP} \
    --amp \
    --recon-path ${RECONPATH}\
    --pretrained ${PRETRAINED}

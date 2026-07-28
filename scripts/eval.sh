#!/bin/bash

MODEL=$1
CACHE=$2
DATADIR=$3
QUERIES=$4
RECON=$5
FTPATH=$6

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --retriever ${MODEL} \
  --cache-corpus ${CACHE} \
  --data-dir ${DATADIR} \
  --queries-path ${QUERIES} \
  --recon_path ${RECON} \
  --ft-model-path ${FTPATH}






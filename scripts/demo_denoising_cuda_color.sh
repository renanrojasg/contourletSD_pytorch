#!/bin/sh
root="./.."
python3 ${root}/demo_denoising_cuda.py \
  --SDmm_dir ${root}/extras/sdmm_matlab \
  --dataset_path ${root}/datasets/BSDS300/images/train \
  --color_images \
  --batch_size 4 \
  --out_dir ${root}/output

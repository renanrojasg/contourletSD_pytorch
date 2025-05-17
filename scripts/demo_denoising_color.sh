#!/bin/sh
root="./.."
python3 ${root}/demo_denoising.py \
  --SDmm_dir ${root}/extras/sdmm_matlab \
  --image_path ${root}/datasets/test_images/peppers_color.png \
  --color_images \
  --out_dir ${root}/output

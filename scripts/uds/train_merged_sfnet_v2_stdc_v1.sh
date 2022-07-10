#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./sfnets/merged_dataset/sfnet_v2_350e_stdc_v1
mkdir -p ${EXP_DIR}
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset merged_dataset \
  --cv 0 \
  --arch network.sfnet_stdcnet.AlignedSTDC1_SFV2_spatial_atten \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1024 \
  --lr 0.02 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --ohem \
  --fpn_dsn_loss \
  --crop_size 1024 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --gblur \
  --max_epoch 350 \
  --wt_bound 1.0 \
  --bs_mult 4 \
  --apex \
  --exp cityscapes_SFsegnet_res18 \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \

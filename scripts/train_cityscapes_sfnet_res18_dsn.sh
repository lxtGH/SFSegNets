#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./sfnets/sfnet_res18_300e_dsn_128_inplane
mkdir -p ${EXP_DIR}
# Example on Cityscapes by resnet50-deeplabv3+ as baseline
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset cityscapes \
  --cv 0 \
  --arch network.sfnet_resnet.DeepR18_SF_deeply_dsn \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1024 \
  --lr 0.01 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --fpn_dsn_loss \
  --crop_size 1024 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --gblur \
  --max_epoch 300 \
  --wt_bound 1.0 \
  --bs_mult 2 \
  --apex \
  --exp cityscapes_SFsegnet_res18 \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt &

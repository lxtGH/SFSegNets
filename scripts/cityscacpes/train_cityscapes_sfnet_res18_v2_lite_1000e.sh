#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./sfnets/sf_r18_v2_spatial_atten_1000e
mkdir -p ${EXP_DIR}
# Example on Cityscapes by resnet50-deeplabv3+ as baseline
# srun -p AD_RD_A100_40G -n1 --gres=gpu:8 --cpus-per-task 8 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 train.py \
  --dataset cityscapes \
  --cv 0 \
  --arch network.sfnet_resnet.DeepR18_SFV2_deeply_dsn_FA_Atten \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1024 \
  --lr 0.01 \
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
  --max_epoch 1000 \
  --wt_bound 1.0 \
  --bs_mult 2 \
  --apex \
  --exp cityscapes_SFsegnet_res18 \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \

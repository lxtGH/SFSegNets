#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
python3 eval.py \
    --dataset cityscapes \
    --arch network.sfnet_resnet.DeepR101_SF_deeply \
    --inference_mode sliding \
    --scales 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
    --split test \
    --cv_split 0 \
    --dump_images \
    --ckpt_path ${2} \
    --snapshot ${1}
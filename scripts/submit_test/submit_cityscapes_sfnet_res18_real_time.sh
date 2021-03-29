#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
python3 eval.py \
    --dataset cityscapes \
    --arch network.sfnet_resnet.DeepR18_SF_deeply \
    --inference_mode whole \
    --scales 1 \
    --split test \
    --cv_split 0 \
    --dump_images \
    --ckpt_path ${2} \
    --snapshot ${1}
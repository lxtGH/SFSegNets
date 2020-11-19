# SFSegNets
Reproduced Implementation of Our ECCV-2020 oral paper: Semantic Flow for Fast and Accurate Scene Parsing.
 
![avatar](./figs/sfnet_res.png)
Our methods achieve the best speed and accuracy trade-off on multiple scene parsing datasets.  

![avatar](./figs/sfnets.png)
Note that the original paper link is on [TorchCV](https://github.com/donnyyou/torchcv) where you can train SFnet models. 
However, that repo is over-complex for further research and exploration.


## DataSet Setting
Please see the DATASETs.md for the details.

## Requirements

pytorch >= 1.2.0
apex
opencv-python

## Pretrained models and Trained CKPTs
Please download the pretrained model including:
resnet18-deep-stem-pytorch:[link](https://drive.google.com/file/d/1AI-VIQPvF9HDsCnpsLwlw1cGnLot5xFD/view?usp=sharing)

dfnetv1:[link](https://drive.google.com/file/d/1xkkmIjKUbMifcrKdWU7I_-Jx_1YQAXfN/view?usp=sharing)

dfnetv2:[link](https://drive.google.com/file/d/1ZRRE99BPhbXwq-ZzO8A5GFmfCe7zxMsz/view?usp=sharing)


and put them into the pretrained_models dir.

Please download the trained model, the mIoU is on Cityscape validation dataset.

resnet18(no-balanced-sample): 78.4 mIoU 

resnet18: 79.0 mIoU [link](https://drive.google.com/file/d/1X7w1HYrSXOJBkfRJuxtXdmR0BXUR-hR8/view?usp=sharing)

resnet50: 80.4 mIoU [link](https://drive.google.com/file/d/1oAOPISp_Rqva_9whsF7eE3pFxuGSc1Wf/view?usp=sharing)

resnet101: 81.2 mIoU [link](https://drive.google.com/file/d/1YPLBTnMit-ybR5pwUhjs-y4KMoT9CvPc/view?usp=sharing)

dfnetv1: 72.2 mIoU [link](https://drive.google.com/file/d/1aP9d4QVbGvBTABOFvi-okOs6DmJU8njH/view?usp=sharing)

dfnetv2: 75.8 mIoU [link](https://drive.google.com/file/d/1iGE9IYImdrs5p0i3k85OoCQzuSUNhjNU/view?usp=sharing)


## Training 

The train settings require 8 GPU with at least 11GB memory.


Train ResNet18 model
```bash
sh ./scripts/train/train_cityscapes_sfnet_res18.sh
```

Train ResNet101 models

```bash
sh ./scripts/train/train_cityscapes_sfnet_res101.sh
```



## Acknowledgement 
This repo is based on Semantic Segmentation from [NVIDIA](https://github.com/NVIDIA/semantic-segmentation) and [DecoupleSegNets](https://github.com/lxtGH/DecoupleSegNets)

Thanks to **SenseTime Research** for Reproducing All these model ckpts and pretrained model.

## Citation
If you find this repo is useful for your research, Please consider citing our paper:


```
@inproceedings{sfnet,
  title={Semantic Flow for Fast and Accurate Scene Parsing},
  author={Li, Xiangtai and You, Ansheng and Zhu, Zhen and Zhao, Houlong and Yang, Maoke and Yang, Kuiyuan and Tong, Yunhai},
  booktitle={ECCV},
  year={2020}
}
```


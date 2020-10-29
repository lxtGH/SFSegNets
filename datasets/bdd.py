"""
BDD Dataset Loader
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import logging
import datasets.uniform as uniform
import datasets.edge_utils as edge_utils
import json
from config import cfg
import datasets.cityscapes_labels as cityscapes_labels

# BDD share the same label map as Cityscapes dataset
trainid_to_name = cityscapes_labels.trainId2name

num_classes = 19
ignore_label = 255
root = cfg.DATASET.BDD_DIR

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def add_items(img_path, mask_path):
    c_items = os.listdir(img_path)
    c_items.sort()
    items = []
    aug_items = []
    for it in c_items:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it.split('.')[0] + "_train_id.png"))
        items.append(item)
    return items, aug_items


def make_dataset(quality, mode):

    items = []
    aug_items = []
    assert quality == 'semantic'
    assert mode in ['train', 'val', 'trainval', 'test']


    img_path = os.path.join(root, 'images', 'train')
    mask_path = os.path.join(root, 'labels','train')

    train_items, train_aug_items = add_items(img_path, mask_path)
    logging.info('BDD has a total of {} train images'.format(len(train_items)))

    img_path = os.path.join(root, 'images', 'val')
    mask_path = os.path.join(root, 'labels', 'val')

    val_items, val_aug_items = add_items(img_path, mask_path, )
    logging.info('BDD has a total of {} validation images'.format(len(val_items)))

    if mode == 'test':
        img_path = os.path.join(root, 'test')
        mask_path = os.path.join(root, 'testannot')
        test_items, test_aug_items = add_items(img_path, mask_path, )
        logging.info('BDD has a total of {} test images'.format(len(test_items)))


    if mode == 'train':
        items = train_items
    elif mode == 'val':
        items = val_items
    elif mode == 'trainval':
        items = train_items + val_items
        aug_items = train_aug_items + val_aug_items
    elif mode == 'test':
        items = test_items
        aug_items = []
    else:
        logging.info('Unknown mode {}'.format(mode))
        sys.exit()

    logging.info('BDD-{}: {} images'.format(mode, len(items)))

    return items, aug_items


class BDD(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=0, class_uniform_tile=0, test=False,
                 cv_split=None, scf=None, hardnm=0, edge_map=False):

        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.scf = scf
        self.hardnm = hardnm
        self.cv_split = cv_split
        self.edge_map = edge_map
        self.centroids = []

        self.imgs, self.aug_imgs = make_dataset(quality, mode)
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for GT data
        if self.class_uniform_pct > 0:
            json_fn = 'bdd_tile{}_cv{}_{}.json'.format(self.class_uniform_tile, self.cv_split, self.mode)

            if os.path.isfile(json_fn):
                with open(json_fn, 'r') as json_data:
                    centroids = json.load(json_data)
                self.centroids = {int(idx): centroids[idx] for idx in centroids}
            else:
                self.centroids = uniform.class_centroids_all(
                        self.imgs,
                        num_classes,
                        id2trainid=None,
                        tile_size=class_uniform_tile)
                with open(json_fn, 'w') as outfile:
                    json.dump(self.centroids, outfile, indent=4)

            self.fine_centroids = self.centroids.copy()

            if self.maxSkip > 0:
                json_fn = 'bdd_tile{}_cv{}_{}_skip{}.json'.format(self.class_uniform_tile, self.cv_split, self.mode, self.maxSkip)
                if os.path.isfile(json_fn):
                    with open(json_fn, 'r') as json_data:
                        centroids = json.load(json_data)
                    self.aug_centroids = {int(idx): centroids[idx] for idx in centroids}
                else:
                    self.aug_centroids = uniform.class_centroids_all(
                            self.aug_imgs,
                            num_classes,
                            id2trainid=None,
                            tile_size=class_uniform_tile)
                    with open(json_fn, 'w') as outfile:
                        json.dump(self.aug_centroids, outfile, indent=4)

                for class_id in range(num_classes):
                    self.centroids[class_id].extend(self.aug_centroids[class_id])

        self.build_epoch()

    def build_epoch(self, cut=False):

        if self.class_uniform_pct > 0:
            if cut:
                self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                self.fine_centroids,
                                                num_classes,
                                                cfg.CLASS_UNIFORM_PCT)
            else:
                self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                self.centroids,
                                                num_classes,
                                                cfg.CLASS_UNIFORM_PCT)
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index):
        elem = self.imgs_uniform[index]
        centroid = None
        if len(elem) == 4:
            img_path, mask_path, centroid, class_id = elem
        else:
            img_path, mask_path = elem

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Image Transformations
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK
                    # We assume that the first transform is capable of taking
                    # in a centroid
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)

        # Debug
        if self.dump_images and centroid is not None:
            outdir = './dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            dump_img_name = trainid_to_name[class_id] + '_' + img_name
            out_img_fn = os.path.join(outdir, dump_img_name + '.png')
            out_msk_fn = os.path.join(outdir, dump_img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.edge_map:
            # _edgemap = np.array(mask_trained)
            # _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)
            _edgemap = mask[:-1, :, :]
            _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)
            edgemap = torch.from_numpy(_edgemap).float()
            return img, mask, edgemap, img_name
        return img, mask, img_name

    def __len__(self):
        return len(self.imgs_uniform)
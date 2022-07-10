"""
IDD Dataset Loader
"""
import logging
import json
import os
import sys
import numpy as np
from PIL import Image
import torch
from torch.utils import data

import torchvision.transforms as transforms
import datasets.uniform as uniform
from config import cfg
import datasets.idd_labels as idd_labels

num_classes = 19
ignore_label = 255
root = cfg.DATASET.IDD_DIR

trainid_to_name = idd_labels.trainId2name
id_to_trainid = idd_labels.label2trainid

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

    folders = os.listdir(img_path)
    items = []
    for f in folders:
        imgs = os.listdir(os.path.join(img_path, f))
        for img in imgs:
            id = img.split('_')[0]
            per_img_path = os.path.join(img_path, f, img)
            mask_name = id + "_" + "gtFine_labelids.png"
            mask_per_path = os.path.join(mask_path, f, mask_name)

            items.append((per_img_path, mask_per_path))
    return items


def make_dataset(quality, mode, maxSkip=0, cv_split=0, hardnm=0):

    assert quality == 'semantic'
    assert mode in ['train', 'val',]

    original_img_dir = "leftImg8bit"
    original_gt_dir = "gtFine"
    img_path = os.path.join(root, original_img_dir, 'train')
    mask_path = os.path.join(root, original_gt_dir, 'train')


    train_items = add_items(img_path, mask_path)
    logging.info('IDD has a total of {} train images'.format(len(train_items)))

    img_path = os.path.join(root, original_img_dir, 'val')
    mask_path = os.path.join(root, original_gt_dir, 'val')

    val_items = add_items(img_path, mask_path)
    logging.info('IDD has a total of {} validation images'.format(len(val_items)))

    if mode == 'train':
        items = train_items
    elif mode == 'val':
        items = val_items
    else:
        logging.info('Unknown mode {}'.format(mode))
        sys.exit()

    logging.info('IDD-{}: {} images'.format(mode, len(items)))

    return items


class IDDUniform(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None, sliding_crop=None,
                 transform=None, target_transform=None, dump_images=False, class_uniform_pct=0.5, class_uniform_tile=1024,
                 test=False, coarse_boost_classes=None, edge_map=False):
        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.coarse_boost_classes = coarse_boost_classes
        self.edge_map = edge_map

        self.imgs = make_dataset(quality, mode, self.maxSkip)
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for fine data
        json_fn = 'idd_{}_tile{}.json'.format(
            self.mode, self.class_uniform_tile)
        if os.path.isfile(json_fn):
            with open(json_fn, 'r') as json_data:
                centroids = json.load(json_data)
            self.centroids = {int(idx): centroids[idx] for idx in centroids}
        else:
            self.centroids = uniform.class_centroids_all(
                self.imgs,
                num_classes,
                id2trainid=id_to_trainid,
                tile_size=class_uniform_tile)
            with open(json_fn, 'w') as outfile:
                json.dump(self.centroids, outfile, indent=4)

        self.fine_centroids = self.centroids.copy()

        self.build_epoch()

    def cities_uniform(self, imgs, name):
        """ list out cities in imgs_uniform """
        cities = {}
        for item in imgs:
            img_fn = item[0]
            img_fn = os.path.basename(img_fn)
            city = img_fn.split('_')[0]
            cities[city] = 1
        city_names = cities.keys()
        logging.info('Cities for {} '.format(name) + str(sorted(city_names)))

    def build_epoch(self, cut=False):
        """
        Perform Uniform Sampling per epoch to create a new list for training such that it
        uniformly samples all classes
        """
        if self.class_uniform_pct > 0:
            if cut:
                # after max_cu_epoch, we only fine images to fine tune
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

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in id_to_trainid.items():
            mask_copy[mask == k] = v
        mask_trained = Image.fromarray(mask_copy.astype(np.uint8))
        mask = mask_trained
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
            outdir = '../../dump_imgs_{}'.format(self.mode)
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


        return img, mask, img_name

    def __len__(self):
        return len(self.imgs_uniform)

    def onehot2label(self, target):
        label = torch.argmax(target, dim=1).long()
        return label
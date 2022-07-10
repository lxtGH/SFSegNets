"""
UDS dataset, Merge Driving Dataset Loader
"""

import os
import sys
import numpy as np
from PIL import Image
from torch.utils import data
import logging
import datasets.uniform as uniform
import json
from config import cfg
import datasets.cityscapes_labels as cityscapes_labels

# Merge Driving share the same label map as Cityscapes dataset
trainid_to_name = cityscapes_labels.trainId2name

num_classes = 19
ignore_label = 255
# set the root of each dataset
bdd_root = cfg.DATASET.BDD_DIR
city_root = cfg.DATASET.CITYSCAPES_DIR
map_root = cfg.DATASET.MAPILLARY_DIR
idd_root = cfg.DATASET.IDD_DIR

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


def add_bdd_items(img_path, mask_path):
    c_items = os.listdir(img_path)
    c_items.sort()
    items = []
    for it in c_items:
        img_per_path = os.path.join(img_path, it)
        mask_per_path = os.path.join(mask_path, it.split('.')[0] + "_train_id.png")
        item = ( img_per_path , mask_per_path)
        items.append(item)
    return items


def add_idd_items(img_path, mask_path):

    folders = os.listdir(img_path)
    items = []
    for f in folders:
        imgs = os.listdir(os.path.join(img_path, f))
        for img in imgs:
            id = img.split('_')[0]
            img_per_path = os.path.join(img_path, f, img)
            mask_name = id + "_" + "gtFine_labelids_19labelTrainIds.png"
            mask_per_path = os.path.join(mask_path, f, mask_name)
            assert os.path.exists(mask_per_path), print(mask_per_path)
            items.append((img_per_path, mask_per_path))
    return items


def add_map_items(img_path, mask_path):
    items = []
    imgs = os.listdir(img_path)
    for img in imgs:
        mask_file = img.split(".")[0] + "_19labelTrainIds.png"
        img_per_path = os.path.join(img_path, img)
        mask_per_path = os.path.join(mask_path, mask_file)
        assert os.path.exists(mask_per_path), print(mask_per_path)
        items.append((img_per_path, mask_per_path))

    return items


def add_city_items(img_path, mask_path):

    citys = os.listdir(img_path)
    items = []
    for f in citys:
        imgs = os.listdir(os.path.join(img_path, f))
        for img in imgs:
            id = "_".join(img.split('_')[:-1])
            img_per_path = os.path.join(img_path, f, img)
            mask_name = id + "_" + "gtFine_labelTrainIds.png"
            mask_per_path = os.path.join(mask_path, f, mask_name)
            assert os.path.exists(mask_per_path), print(mask_per_path)
            items.append((img_per_path, mask_per_path))
    return items


def make_dataset(quality, mode):

    assert quality == 'semantic'
    assert mode in ['train', 'val', 'trainval', 'test']

    train_items = []
    val_items = []

    # add bdd datasets
    img_path = os.path.join(bdd_root, 'images', 'train')
    mask_path = os.path.join(bdd_root, 'labels', 'train')

    bdd_train_items = add_bdd_items(img_path, mask_path)
    logging.info('BDD has a total of {} train images'.format(len(bdd_train_items)))
    train_items.extend(bdd_train_items)

    img_path = os.path.join(bdd_root, 'images', 'val')
    mask_path = os.path.join(bdd_root, 'labels', 'val')

    bdd_val_items = add_bdd_items(img_path, mask_path, )
    logging.info('BDD has a total of {} validation images'.format(len(bdd_val_items)))
    val_items.extend(bdd_val_items)

    # add mapillary dataset

    map_img_path = os.path.join(map_root, "training", "images")
    map_mask_path = os.path.join(map_root, "training", "labels")
    map_train_items = add_map_items(map_img_path, map_mask_path)
    logging.info('Mapillary has a total of {} train images'.format(len(map_train_items)))
    train_items.extend(map_train_items)

    map_img_path = os.path.join(map_root, "validation", "images")
    map_mask_path = os.path.join(map_root, "validation", "labels")
    map_val_items = add_map_items(map_img_path, map_mask_path)
    logging.info('Mapillary has a total of {} validation images'.format(len(map_val_items)))
    val_items.extend(map_val_items)

    # add IDD dataset
    idd_img_dir = "leftImg8bit"
    idd_gt_dir = "gtFine"
    img_path = os.path.join(idd_root, idd_img_dir, 'train')
    mask_path = os.path.join(idd_root, idd_gt_dir, 'train')

    idd_train_items = add_idd_items(img_path, mask_path)
    logging.info('IDD has a total of {} train images'.format(len(idd_train_items)))
    train_items.extend(idd_train_items)

    img_path = os.path.join(idd_root, idd_img_dir, 'val')
    mask_path = os.path.join(idd_root, idd_gt_dir, 'val')

    idd_val_items = add_idd_items(img_path, mask_path)
    logging.info('IDD has a total of {} validation images'.format(len(val_items)))
    val_items.extend(idd_val_items)

    # add cityscapes fine dataset
    city_img_dir = "leftImg8bit_trainvaltest/leftImg8bit"
    city_mask_dir = "gtFine_trainvaltest/gtFine"

    img_path = os.path.join(city_root, city_img_dir, 'train')
    mask_path = os.path.join(city_root,city_mask_dir, "train")
    city_train_items = add_city_items(img_path, mask_path)
    logging.info('CityScapes has a total of {} train images'.format(len(city_train_items)))
    train_items.extend(city_train_items)

    img_path = os.path.join(city_root, city_img_dir, 'val')
    mask_path = os.path.join(city_root, city_mask_dir, "val")
    city_val_items = add_city_items(img_path, mask_path)
    logging.info('CityScapes has a total of {} validation images'.format(len(city_val_items)))
    val_items.extend(city_val_items)


    if mode == 'train':
        items = train_items
    elif mode == 'val':
        items = val_items
    else:
        logging.info('Unknown mode {}'.format(mode))
        sys.exit()

    logging.info('Merged Dataset have -{}: {} images for {}'.format(mode, len(items), mode))

    return items


class MergeDrivingDataset(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=0.5, class_uniform_tile=1024, test=False,
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

        self.imgs = make_dataset(quality, mode)
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for GT data
        if self.class_uniform_pct > 0:
            json_fn = 'merge_drive_tile{}_cv{}_{}.json'.format(self.class_uniform_tile, self.cv_split, self.mode)

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

        return img, mask, img_name

    def __len__(self):
        return len(self.imgs_uniform)
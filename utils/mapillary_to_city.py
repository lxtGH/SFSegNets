import argparse
import os.path as osp

import mmcv
import numpy as np
from PIL import Image


def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    id_to_trainid = {
        13: 0,
        24: 0,
        41: 0,
        2:  1,
        15: 1,
        17: 2,
        6:  3,
        3:  4,
        45: 5,
        47: 5,
        48: 6,
        50: 7,
        30: 8,
        29: 9,
        27: 10,
        19: 11,
        20: 12,
        21: 12,
        22: 12,
        55: 13,
        61: 14,
        54: 15,
        58: 16,
        57: 17,
        52: 18
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_file = file.replace('.png', '_19labelTrainIds.png')
    assert file != new_file
    # new_file_name = new_file.split("/")[-1]
    # new_file = os.path.join(out_dir, new_file_name)
    Image.fromarray(label_copy, mode='L').save(new_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary annotations to Cityscapes style TrainIds')
    parser.add_argument('map_path', help='map data path')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    map_path = args.map_path

    gt_dir = osp.join(map_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix=tuple(f'.png'),
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)

    if args.nproc > 1:
        sample_class_stats = mmcv.track_parallel_progress(
            convert_to_train_id, poly_files, args.nproc)
    else:
        sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                 poly_files)
    print("Finished!")


if __name__ == '__main__':
    main()

#----Data-Structure
# mapillary
#  └ images
#    └ train
#    └ val
#  └ labels
#    └ train
#    └ val
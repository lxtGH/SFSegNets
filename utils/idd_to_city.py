import argparse
import os.path as osp

import mmcv
import numpy as np
from PIL import Image
from datasets.idd_labels import label2trainid


def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    id_to_trainid = label2trainid
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
    Image.fromarray(label_copy, mode='L').save(new_file)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary annotations to Cityscapes style TrainIds')
    parser.add_argument('idd_path', help='idd data path')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    idd_path = args.idd_path

    gt_dir = osp.join(idd_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix='.png',
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)
    print("Found ", len(poly_files), "images")
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
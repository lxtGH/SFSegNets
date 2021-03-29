import os
import logging
import time
import argparse
import datetime
from PIL import Image
import numpy as np
import cv2


import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

from network import get_net
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg
from utils.misc import save_log

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--demo_folder', type=str, default='', help='path to the folder containing demo images')
parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint')
parser.add_argument('--arch', type=str, default='network.sfnet_resnet.DeepR18_SF_deeply_dsn', help='network architecture used for inference')
parser.add_argument('--save_dir', type=str, default='./save', help='path to save your results')
args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

# setup logger
date_str = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
log_dir = os.path.join(args.save_dir, "log")
os.makedirs(log_dir, exist_ok=True)
save_log('log', log_dir, date_str, rank=0)

# get net
args.dataset_cls = cityscapes
net = get_net(args, criterion=None)
net = torch.nn.DataParallel(net).cuda()
logging.info('Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
net.eval()
logging.info('Net restored.')

# get data
data_dir = args.demo_folder
images = os.listdir(data_dir)
if len(images) == 0:
    logging.info('There are no images at directory %s. Check the data path.' % (data_dir))
else:
    logging.info('There are %d images to be processed.' % (len(images)))
images.sort()

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

start_time = time.time()
for img_id, img_name in enumerate(images):
    img_dir = os.path.join(data_dir, img_name)
    img = Image.open(img_dir).convert('RGB')
    img_tensor = img_transform(img)

    # predict
    with torch.no_grad():
        pred = net(x=img_tensor.unsqueeze(0).cuda())
        logging.info('%04d/%04d: Inference done.' % (img_id + 1, len(images)))

    # final mask
    pred = pred.cpu().numpy().squeeze()
    pred = np.argmax(pred, axis=0)

    # final mask
    color_name = 'color_mask_' + img_name
    overlap_name = 'overlap_' + img_name

    # save colorized predictions
    colorized = args.dataset_cls.colorize_mask(pred)
    colorized.save(os.path.join(args.save_dir, color_name))

    # save colorized predictions overlapped on original images
    overlap = cv2.addWeighted(np.array(img), 0.5, np.array(colorized.convert('RGB')), 0.5, 0)
    cv2.imwrite(os.path.join(args.save_dir, overlap_name), overlap[:, :, ::-1])


end_time = time.time()

logging.info('Results saved.')
logging.info('Inference takes %4.2f seconds, which is %4.2f seconds per image, including saving results.' % (end_time - start_time, (end_time - start_time)/len(images)))

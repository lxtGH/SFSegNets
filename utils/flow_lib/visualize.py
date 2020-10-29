"""
# ==============================
# from flowlib.py
# library for optical flow processing
# Author: Ruoteng Li
# Date: 6th Aug 2016
# ==============================
"""
import os
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
import cv2


from sklearn.decomposition import PCA
from .io import read_flow
from .img import flow_to_image
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def visualize_img(img):
    """
    visualize the images using matplotlib
    :param img: input image array()
    :return: None
    """
    plt.imshow(img)
    plt.show()
    return None


def show_flow(filename):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file (.flo file)
    :return: None
    """
    flow = read_flow(filename)
    img = flow_to_image(flow)
    plt.imshow(img)
    plt.show()

def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow)
        plt.imshow(img)
        plt.show()
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = max(np.max(du), np.max(dv))
        img = np.zeros((h, w, 3), dtype=np.float64)
        # angle layer
        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid
        # show
        plt.imshow(img)
        plt.show()
    return None


def segment_flow(flow):

    h = flow.shape[0]
    w = flow.shape[1]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idx = ((abs(u) > LARGEFLOW) | (abs(v) > LARGEFLOW))
    idx2 = (abs(u) == SMALLFLOW)
    class0 = (v == 0) & (u == 0)
    u[idx2] = 0.00001
    tan_value = v / u

    class1 = (tan_value < 1) & (tan_value >= 0) & (u > 0) & (v >= 0)
    class2 = (tan_value >= 1) & (u >= 0) & (v >= 0)
    class3 = (tan_value < -1) & (u <= 0) & (v >= 0)
    class4 = (tan_value < 0) & (tan_value >= -1) & (u < 0) & (v >= 0)
    class8 = (tan_value >= -1) & (tan_value < 0) & (u > 0) & (v <= 0)
    class7 = (tan_value < -1) & (u >= 0) & (v <= 0)
    class6 = (tan_value >= 1) & (u <= 0) & (v <= 0)
    class5 = (tan_value >= 0) & (tan_value < 1) & (u < 0) & (v <= 0)

    seg = np.zeros((h, w))

    seg[class1] = 1
    seg[class2] = 2
    seg[class3] = 3
    seg[class4] = 4
    seg[class5] = 5
    seg[class6] = 6
    seg[class7] = 7
    seg[class8] = 8
    seg[class0] = 0
    seg[idx] = 0

    return seg


def draw_flow(img, flow, step=32):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_flow_arrow(img, flow, id, step=32):
    h, w = img.shape[:2]
    # print(img.shape)
    # print(flow.shape)
    # img = np.ones((h, w, 3)) * 255

    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    fx, fy = fx *(id+2), fy*(id+2)
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = img
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # vis = img
    # cv2.arrowedLine(vis,(y,x),(y+fy,x+fx),(255,0,0) )
    # cv2.polylines(vis, lines, 0, (255, 0, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis,(x2,y2),(x1,y1), (255,0,0),thickness=2, tipLength=0.2)
    return vis


def draw_flow_arrow_white(img, flow, id, step=32):
    h, w = img.shape[:2]
    # print(img.shape)
    # print(flow.shape)
    img = np.ones((h, w, 3)) * 255

    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    fx, fy = fx *(id+2), fy*(id+2)
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    vis = img
    # cv2.arrowedLine(vis,(y,x),(y+fy,x+fx),(255,0,0) )
    # cv2.polylines(vis, lines, 0, (255, 0, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis,(x2,y2),(x1,y1), (255,0,0),thickness=2, tipLength=0.2)
    return vis



def visulizeFeatureMap(feature, name, out_path="/home/lxt/debug_two_path"):
    feature = feature.data.cpu().numpy()
    img_out = np.mean(feature, axis=0)
    cv2.normalize(img_out, img_out, 0, 255, cv2.NORM_MINMAX)
    img_out = np.array(img_out, dtype=np.uint8)
    img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    if os.path.exists(out_path) == False:
        os.makedirs(out_path)
    file_name = os.path.join(out_path, name)
    cv2.imwrite(file_name, img_out)


def visulizeFeatureMapSingle(feature, name, out_path="/home/lxt/debug_two_path"):
    feature = feature.squeeze().data.cpu().numpy()
    img_out = feature
    cv2.normalize(img_out, img_out, 0, 255, cv2.NORM_MINMAX)
    img_out = np.array(img_out, dtype=np.uint8)
    img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    if os.path.exists(out_path) == False:
        os.makedirs(out_path)
    file_name = os.path.join(out_path, name)
    cv2.imwrite(file_name, img_out)


def visulizeFeatureMapPCA(feature, name, out_path="/home/lxt/debug_two_path"):
    if os.path.exists(out_path) == False:
      os.makedirs(out_path)
    feature = feature.squeeze().data.cpu().numpy()
    # img_out = np.mean(feature, axis=0)
    c, h, w = feature.shape
    img_out = feature.reshape(c, -1).transpose(1, 0)
    pca = PCA(n_components=3)
    pca.fit(img_out)
    img_out_pca = pca.transform(img_out)
    img_out_pca = img_out_pca.transpose(1, 0).reshape(3, h, w).transpose(1, 2, 0)

    cv2.normalize(img_out_pca, img_out_pca, 0, 255, cv2.NORM_MINMAX)
    img_out_pca = cv2.resize(img_out_pca, (2048, 1024),interpolation=cv2.INTER_LINEAR)
    img_out = np.array(img_out_pca, dtype=np.uint8)
    # img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
    file_name = os.path.join(out_path, name)
    cv2.imwrite(file_name, img_out)
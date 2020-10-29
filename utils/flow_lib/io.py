import numpy as np
import png
# from stackoverflow
# http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
TAG_CHAR = np.array([202021.25], np.float32)

"""
==============
Read and Write Section
==============
"""


def read_flow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file")
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def write_flow(filename, uv, v=None):
    """ Write optical flow to file.
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2
    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv
    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

"""
==============
Quantization Section
==============
"""


def quantize_flow(flow, max_val=0.02, norm=True):
    """Quantize flow to [0, 255] (much smaller size when dumping as images)
    Args:
        flow(ndarray): optical flow
        max_val(float): maximum value of flow, values beyond
                        [-max_val, max_val] will be truncated.
        norm(bool): whether to divide flow values by width/height
    Returns:
        tuple: quantized dx and dy
    """
    h, w, _ = flow.shape
    dx = flow[..., 0]
    dy = flow[..., 1]
    if norm:
        dx = dx / w  # avoid inplace operations
        dy = dy / h
    dx = np.maximum(0, np.minimum(dx + max_val, 2 * max_val))
    dy = np.maximum(0, np.minimum(dy + max_val, 2 * max_val))
    dx = np.round(dx * 255 / (max_val * 2)).astype(np.uint8)
    dy = np.round(dy * 255 / (max_val * 2)).astype(np.uint8)
    return dx, dy


def dequantize_flow(dx, dy, max_val=0.02, denorm=True):
    """Recover flow from quantized flow
    Args:
        dx(ndarray): quantized dx
        dy(ndarray): quantized dy
        max_val(float): maximum value used when quantizing.
        denorm(bool): whether to multiply flow values with width/height
    Returns:
        tuple: dequantized dx and dy
    """
    assert dx.shape == dy.shape
    assert dx.ndim == 2 or (dx.ndim == 3 and dx.shape[-1] == 1)
    dx = dx.astype(np.float32) * max_val * 2 / 255 - max_val
    dy = dy.astype(np.float32) * max_val * 2 / 255 - max_val
    if denorm:
        dx *= dx.shape[1]
        dy *= dx.shape[0]
    flow = np.dstack((dx, dy))
    return flow



"""
==============
Disparity Section
==============
"""

def read_disp_png(file_name):
    """
    Read optical flow from KITTI .png file
    :param file_name: name of the flow file
    :return: optical flow data in matrix
    """
    image_object = png.Reader(filename=file_name)
    image_direct = image_object.asDirect()
    image_data = list(image_direct[2])
    (w, h) = image_direct[3]['size']
    channel = len(image_data[0]) / w
    flow = np.zeros((h, w, channel), dtype=np.uint16)
    for i in range(len(image_data)):
        for j in range(channel):
            flow[i, :, j] = image_data[i][j::channel]
    return flow[:, :, 0] / 256


def disp_to_flowfile(disp, filename):
    """
    Read KITTI disparity file in png format
    :param disp: disparity matrix
    :param filename: the flow file name to save
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = disp.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    empty_map = np.zeros((height, width), dtype=np.float32)
    data = np.dstack((disp, empty_map))
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    data.tofile(f)
    f.close()

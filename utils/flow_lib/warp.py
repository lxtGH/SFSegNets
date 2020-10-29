import torch
import torch.nn.functional as F
from torch.autograd import Variable


def flow_warp(image,flow,direction = 1,mode="bilinear"):
    """
    :param image: input image for warp ,size: (b,c,h,w)
    :param flow: dense flow field  ,size: (b,2,h,w)
    :param direction: determine the direction of the flow
    :return: warped image ,size: (b,c,h,w)
    """
    n,c,h,w = image.size()
    x = Variable(torch.arange(0,w).expand([n,h,w])).cuda() #(n,h,w)
    y = Variable(torch.arange(0,h).expand([n,w,h])).permute(0,2,1).cuda()

    grid_x = (x + flow[:,0,:,:] * direction) * 2 /(w - 1) - 1.0
    grid_y = (y + flow[:,1,:,:] * direction) * 2 /(h - 1) - 1.0

    grid_xy = torch.cat([torch.unsqueeze(grid_x, 3), torch.unsqueeze(grid_y, 3)],3) # c,h,w,n

    warp_image = F.grid_sample(image, grid_xy, mode)

    return warp_image


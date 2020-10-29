import numpy as np
import cv2


def seg2bmap(seg, width=None, height=None):
	"""
	From a segmentation, compute a binary boundary map with 1 pixel wide
	boundaries.  The boundary pixels are offset by 1/2 pixel towards the
	origin from the actual segment boundary.

	Arguments:
		seg     : Segments labeled from 1..k.
		width	  :	Width of desired bmap  <= seg.shape[1]
		height  :	Height of desired bmap <= seg.shape[0]

	Returns:
		bmap (ndarray):	Binary boundary map.

	 David Martin <dmartin@eecs.berkeley.edu>
	 January 2003
 """

	seg = seg.astype(np.bool)
	seg[seg>0] = 1

	assert np.atleast_3d(seg).shape[2] == 1

	width  = seg.shape[1] if width  is None else width
	height = seg.shape[0] if height is None else height

	h,w = seg.shape[:2]

	ar1 = float(width) / float(height)
	ar2 = float(w) / float(h)

	assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
			'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

	e  = np.zeros_like(seg)
	s  = np.zeros_like(seg)
	se = np.zeros_like(seg)

	e[:,:-1]    = seg[:,1:]
	s[:-1,:]    = seg[1:,:]
	se[:-1,:-1] = seg[1:,1:]

	b        = seg^e | seg^s | seg^se
	b[-1,:]  = seg[-1,:]^e[-1,:]
	b[:,-1]  = seg[:,-1]^s[:,-1]
	b[-1,-1] = 0

	if w == width and h == height:
		bmap = b
	else:
		bmap = np.zeros((height,width))
		for x in range(w):
			for y in range(h):
				if b[y,x]:
					j = 1+np.floor((y-1)+height / h)
					i = 1+np.floor((x-1)+width  / h)
					bmap[j,i] = 1
	return bmap


if __name__ == '__main__':
	img = cv2.imread("/home/SENSETIME/lixiangtai/data/test_edge/frankfurt_000000_009969_gtFine_labelIds.png", 0)
	print(img)
	# img += 1
	print(np.unique(img))
	bit_map = seg2bmap(img)
	print(bit_map)
	bit_map = np.array(bit_map, dtype=np.int8)
	print(np.unique(bit_map), "boundary:", np.sum(bit_map))
	cv2.imwrite("/home/SENSETIME/lixiangtai/data/test_edge/frankfurt_000000_009969_gtFine_labelIds_edge.png",bit_map)
	bit_map_color = np.zeros((1024,2048,3))
	bit_map_color[bit_map == 1] = [255,255,255]
	cv2.imwrite("/home/SENSETIME/lixiangtai/data/test_edge/frankfurt_000000_009969_gtFine_labelIds_edge_color.png", bit_map_color)
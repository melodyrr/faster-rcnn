import numpy as np



# 这个函数的作用就是产生(0,0)坐标开始的基础的9个anchor框
# 分析一下函数的参数base_size=16就是基础的anchor的宽和高其实是16的大小
# 再根据不同的放缩比和宽高比进行进一步的调整
# ratios就是指的宽高的放缩比分别是0.5:1,1:1,1:2
# 最后一个参数是anchor_scales也就是在base_size的基础上再增加的量，
# 本代码中对应着三种面积的大小(16*8)^2 (16*16)^2 (16*32)^2  
# 也就是128,256,512的平方大小，三种面积乘以三种放缩比就刚刚好是9种anchor
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    """Generate anchor base windows by enumerating aspect ratio and scales.
        (y_{min}, x_{min}, y_{max}, x_{max}) of a bounding box."""
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


# 根据上面函数在0,0生成的anchor_base,再原图的所有点上生成anchors
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):

    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


if __name__ == "__main__":
	anchors = generate_anchor_base()
	print(anchors.shape)
	#print(anchors[0])
	mode = 1

	# ratio = 宽/高
	from config import opt
	from dataset import Dataset, inverse_normalize
	import torch
	dataset = Dataset(opt)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
	img, bbox, label, scale = next(iter(dataloader))

	import matplotlib.pyplot as plt
	if mode == 0:
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(inverse_normalize(img[0].numpy()).transpose((1, 2, 0)).astype(np.uint8))
		ax.set_xlim([-500,1300])
		ax.set_ylim([1300,-500])
		for i in range(9):
			y1 = anchors[i][0]
			x1 = anchors[i][1]
			y2 = anchors[i][2]
			x2 = anchors[i][3]
			height = y2 - y1
			width = x2 - x1
			if i in [0,1,2]:
				ax.add_patch(plt.Rectangle((x1,y1), width, height, fill=False, edgecolor='red', linewidth=1))
			if i in [3,4,5]:
				ax.add_patch(plt.Rectangle((x1,y1), width, height, fill=False, edgecolor='green', linewidth=1))
			if i in [6,7,8]:
				ax.add_patch(plt.Rectangle((x1,y1), width, height, fill=False, edgecolor='blue', linewidth=1))
		plt.show()
	else:
		# 600,800 = img.shape[2:]
		# 16 = feat_stride
		all_anchor = _enumerate_shifted_anchor(np.array(anchors),16, 37, 50)
		print(all_anchor.shape)  # 4320000 = 600*800*9
		# ratio = 宽/高
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(inverse_normalize(img[0].numpy()).transpose((1, 2, 0)).astype(np.uint8))
		ax.set_xlim([-500,1500])
		ax.set_ylim([1500,-100])

		# 等了三分钟画不出4320000个。。。少画一点吧。

		for i in range(-18,-9):
		    y1 = all_anchor[i][0]
		    x1 = all_anchor[i][1]
		    y2 = all_anchor[i][2]
		    x2 = all_anchor[i][3]
		    height = y2 - y1
		    width = x2 - x1
		    print(x1+width//2, y1+height//2)
		    if i%9 in [0,1,2]:
		        ax.add_patch(plt.Rectangle((x1,y1), width, height, fill=False, edgecolor='red', linewidth=1))
		    if i%9 in [3,4,5]:
		        ax.add_patch(plt.Rectangle((x1,y1), width, height, fill=False, edgecolor='green', linewidth=1))
		    if i%9 in [6,7,8]:
		        ax.add_patch(plt.Rectangle((x1,y1), width, height, fill=False, edgecolor='blue', linewidth=1))
		plt.show()

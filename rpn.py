import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from anchor import generate_anchor_base, _enumerate_shifted_anchor
from creator_tool import ProposalCreator





'''
RPN网络的输入为图像特征。RPN网络是全卷积网络。RPN网络要完成的任务是训练自己、提供rois。

训练自己：二分类、bounding box 回归（由AnchorTargetCreator实现）

提供rois：为Fast-RCNN提供训练需要的rois（由ProposalCreator实现）

RPN网络流程我们之前介绍过一部分，这里完整的实现整个网络。

首先初始化网络的结构：特征（N，512，h，w）输入进来（原图像的大小：16h，16w），首先是加pad的512个33大小卷积核，输出仍为（N，512，h，w）。然后左右两边各有一个11卷积。左路为18个11卷积，输出为（N，18，h，w），即所有anchor的0-1类别概率（hw约为2400，hw9约为20000）。右路为36个1*1卷积，输出为（N，36，h，w），即所有anchor的回归位置参数。

前向传播：输入特征即feature map，调用函数_enumerate_shifted_anchor生成全部20000个anchor。然后特征经过卷积，在经过两路卷积分别输出rpn_locs, rpn_scores。然后rpn_locs, rpn_scores作为ProposalCreator的输入产生2000个rois，同时还有 roi_indices，这个 roi_indices在此代码中是多余的，因为我们实现的是batch_siae=1的网络，一个batch只会输入一张图象。如果多张图象的话就需要存储索引以找到对应图像的roi。
'''
class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN."""

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict()):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network."""
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor



# 用来为权重初始化
def normal_init(m, mean, stddev, truncated=False):
    """weight initalizer: truncated normal and random normal."""
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()






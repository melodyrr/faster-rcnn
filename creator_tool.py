import numpy as np
import torch
#from torchvision.ops import nms
from nms import nms
from iou import bbox_iou
from boxcoder import bbox2loc, loc2bbox




# 目的：利用每张图中bbox的真实标签来为所有任务分配ground truth！
# 输入：最初生成的20000个anchor坐标、此一张图中所有的bbox的真实坐标
# 输出：size为（20000，1）的正负label（其中只有128个为1，128个为0，其余都为-1
#       size为（20000，4）的回归目标（所有anchor的坐标都有）

# 将20000多个候选的anchor选出256个anchor进行二分类和所有的anchor进行回归位置 。为上面的预测值提供相应的真实值。选择方式如下：
# 对于每一个ground truth bounding box (gt_bbox)，选择和它重叠度（IoU）最高的一个anchor作为正样本。
# 对于剩下的anchor，从中选择和任意一个gt_bbox重叠度超过0.7的anchor，作为正样本，正样本的数目不超过128个。
# 随机选择和gt_bbox重叠度小于0.3的anchor作为负样本。负样本和正样本的总数为256。
# 对于每个anchor, gt_label 要么为1（前景），要么为0（背景），所以这样实现二分类。
# 在计算回归损失的时候，只计算正样本（前景）的损失，不计算负样本的位置损失。
class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors."""

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors."""

        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        # argmax_ious [S,] 每个Anchor对应iou最大的bbox是第几个
        # max_ious 每个anchor对应iou最大bbox的iou值
        # gt_argmax_ious 对于每一个(gt_bbox)，和它IoU最高的一个anchor的位置
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)


        # 如果每个anchor对应bbox的最大iou值都小于0.3,那么这就是负样本。
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        # 正样本不能太多
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        # 负样本也不能太多，正+负=256。正的不够128，负的来凑。
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


# 把除掉的不完全在图片以内的anchors拿回来。
def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


# 计算哪些Anchor完全落在图片内部。
def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside



'''
RPN利用 AnchorTargetCreator自身训练的同时，还会提供RoIs（region of interests）给Fast RCNN（RoIHead）作为训练样本。RPN生成RoIs的过程(ProposalCreator)如下：

对于每张图片，利用它的feature map， 计算 (H/16)× (W/16)×9（大概20000）个anchor属于前景的概率，以及对应的位置参数。选取概率较大的12000个anchor利用回归的位置参数，修正这12000个anchor的位置，得到RoIs。

利用非极大值（(Non-maximum suppression, NMS）抑制，选出概率最大的2000个RoIs

注意：在inference的时候，为了提高处理速度，12000和2000分别变为6000和300.

注意：这部分的操作不需要进行反向传播，因此可以利用numpy/tensor实现。

RPN的输出：RoIs（形如2000×4或者300×4的tensor）
'''
# 目的：为Fast-RCNN也即检测网络提供2000个训练样本
# 输入：RPN网络中1*1卷积输出的loc和score，以及20000个anchor坐标，原图尺寸，scale（即对于这张训练图像较其原始大小的scale）
# 输出：2000个训练样本rois（只是2000*4的坐标，无ground truth！）
class ProposalCreator:

    """Proposal regions are generated by calling this object.

        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes."""

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        """Propose RoIs."""
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        rois = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        # y小于图片高度
        rois[:, slice(0, 4, 2)] = np.clip(
            rois[:, slice(0, 4, 2)], 0, img_size[0])
        # x小于图片宽度
        rois[:, slice(1, 4, 2)] = np.clip(
            rois[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale
        hs = rois[:, 2] - rois[:, 0]
        ws = rois[:, 3] - rois[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        rois = rois[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        rois= rois[order, :]
        score = score[order]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        keep = nms(rois, score, self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        rois = rois[keep]
        return rois





'''
ProposalTargetCreator是RPN网络与ROIHead网络的过渡操作，前面讲过，RPN会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，而是利用ProposalTargetCreator 选择128个RoIs用以训练。选择的规则如下：

RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个） 选择 RoIs和gt_bboxes的IoU小于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本 为了便于训练，对选择出的128个RoIs，还对他们的gt_roi_loc 进行标准化处理（减去均值除以标准差）

对于分类问题,直接利用交叉熵损失. 而对于位置的回归损失,一样采用Smooth_L1Loss, 只不过只对正样本计算损失.而且是只对正样本中的这个类别4个参数计算损失。举例来说:

一个RoI在经过FC 84后会输出一个84维的loc 向量. 如果这个RoI是负样本,则这84维向量不参与计算 L1_Loss 如果这个RoI是正样本,属于label K,那么它的第 K×4, K×4+1 ，K×4+2， K×4+3 这4个数参与计算损失，其余的不参与计算损失。
'''

# 目的：为2000个rois赋予ground truth！（严格讲挑出128个赋予ground truth！）
# 输入：2000个rois、一个batch（一张图）中所有的bbox ground truth（R，4）、对应bbox所包含的label（R，1）（VOC2007来说20类0-19）
# 输出：128个sample roi（128，4）、128个gt_roi_loc（128，4）、128个gt_roi_label（128，1）

class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs."""

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals.

           sample_roi: Regions of interests that are sampled. Its shape is :math:`(S, 4)`.
           gt_roi_loc: Offsets and scales to match the sampled RoIs to the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
           gt_roi_label: Labels assigned to sampled RoIs. Its shape is math:`(S,)`. Its range is :math:`[0, L]`. The label with value 0 is the background."""

        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label

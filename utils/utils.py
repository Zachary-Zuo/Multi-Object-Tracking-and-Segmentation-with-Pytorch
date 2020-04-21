import glob
import random
import time
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    # x, y are coordinates of center
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively.
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

@torch.jit.script
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd:bool=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)

def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    # x, y are coordinates of center
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively.
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)  # Bottom left x
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)  # Bottom left y
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)  # Top right x
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)  # Top right y
    return y

def fast_nms(boxes, scores, iou_thres: float = 0.5, top_k: int = 200, second_threshold: bool = False,
             conf_thres: float = 0.5):
    '''
    Vectorized, approximated, fast NMS, adopted from YOLACT:
    https://github.com/dbolya/yolact/blob/master/layers/functions/detection.py
    The original version is for multi-class NMS, here we simplify the code for single-class NMS
    '''
    scores, idx = scores.sort(0, descending=True)

    idx = idx[:top_k].contiguous()
    scores = scores[:top_k]
    num_dets = idx.size()

    boxes = boxes[idx, :]

    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=0)

    keep = (iou_max <= iou_thres)

    if second_threshold:
        keep *= (scores > self.conf_thresh)

    return idx[keep]

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, method='standard'):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Args:
        prediction,
        conf_thres,
        nms_thres,
        method = 'standard' or 'fast'
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Non-maximum suppression
        if method == 'standard':
            nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == 'fast':
            nms_indices = fast_nms(pred[:, :4], pred[:, 4], iou_thres=nms_thres, conf_thres=conf_thres)
        else:
            raise ValueError('Invalid NMS type!')
        det_max = pred[nms_indices]

        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output
import os
from collections import defaultdict,OrderedDict

import torch.nn as nn
import torch

import time
import torch.nn.functional as F
import numpy as np
import math

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, nID, nE, img_size, yolo_layer):
        super(YOLOLayer, self).__init__()
        self.layer = yolo_layer
        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.nID = nID  # number of identities
        self.img_size = 0
        self.emb_dim = nE
        self.shift = [1, 3, 5]

        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.SoftmaxLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.s_c = nn.Parameter(-4.15 * torch.ones(1))  # -4.15
        self.s_r = nn.Parameter(-4.85 * torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3 * torch.ones(1))  # -2.3

        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1) if self.nID > 1 else 1

    def forward(self, p_cat, img_size, targets=None, classifier=None, test_emb=False):
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        nB, nGh, nGw = p.shape[0], p.shape[-2], p.shape[-1]

        if self.img_size != img_size:
            create_grids(self, img_size, nGh, nGw)

            if p.is_cuda:
                self.grid_xy = self.grid_xy.cuda()
                self.anchor_wh = self.anchor_wh.cuda()

        p = p.view(nB, self.nA, self.nC + 5, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        p_emb = p_emb.permute(0, 2, 3, 1).contiguous()
        p_box = p[..., :4]
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # Conf

        # Training
        if targets is not None:
            if test_emb:
                tconf, tbox, tids = build_targets_max(targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw)
            else:
                tconf, tbox, tids = build_targets_thres(targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw)
            tconf, tbox, tids = tconf.cuda(), tbox.cuda(), tids.cuda()
            mask = tconf > 0

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nP = torch.ones_like(mask).sum().float()
            if nM > 0:
                lbox = self.SmoothL1Loss(p_box[mask], tbox[mask])
            else:
                FT = torch.cuda.FloatTensor if p_conf.is_cuda else torch.FloatTensor
                lbox, lconf = FT([0]), FT([0])
            lconf = self.SoftmaxLoss(p_conf, tconf)
            lid = torch.Tensor(1).fill_(0).squeeze().cuda()
            emb_mask, _ = mask.max(1)

            # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
            tids, _ = tids.max(1)
            tids = tids[emb_mask]
            embedding = p_emb[emb_mask].contiguous()
            embedding = self.emb_scale * F.normalize(embedding)
            nI = emb_mask.sum().float()

            if test_emb:
                if np.prod(embedding.shape) == 0 or np.prod(tids.shape) == 0:
                    return torch.zeros(0, self.emb_dim + 1).cuda()
                emb_and_gt = torch.cat([embedding, tids.float()], dim=1)
                return emb_and_gt

            if len(embedding) > 1:
                logits = classifier(embedding).contiguous()
                lid = self.IDLoss(logits, tids.squeeze())

            # Sum loss components
            loss = torch.exp(-self.s_r) * lbox + torch.exp(-self.s_c) * lconf + torch.exp(-self.s_id) * lid + \
                   (self.s_r + self.s_c + self.s_id)
            loss *= 0.5

            return loss, loss.item(), lbox.item(), lconf.item(), lid.item(), nT

        else:
            p_conf = torch.softmax(p_conf, dim=1)[:, 1, ...].unsqueeze(-1)
            p_emb = F.normalize(p_emb.unsqueeze(1).repeat(1, self.nA, 1, 1, 1).contiguous(), dim=-1)
            # p_emb_up = F.normalize(shift_tensor_vertically(p_emb, -self.shift[self.layer]), dim=-1)
            # p_emb_down = F.normalize(shift_tensor_vertically(p_emb, self.shift[self.layer]), dim=-1)
            p_cls = torch.zeros(nB, self.nA, nGh, nGw, 1).cuda()  # Temp
            p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)
            # p = torch.cat([p_box, p_conf, p_cls, p_emb, p_emb_up, p_emb_down], dim=-1)
            p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.to(p))
            p[..., :4] *= self.stride

            return p.view(nB, -1, p.shape[-1])

def create_grids(self, img_size, nGh, nGw):
    self.stride = img_size[0]/nGw
    assert self.stride == img_size[1] / nGh, \
            "{} v.s. {}/{}".format(self.stride, img_size[1], nGh)

    # build xy offsets
    grid_x = torch.arange(nGw).repeat((nGh, 1)).view((1, 1, nGh, nGw)).float()
    grid_y = torch.arange(nGh).repeat((nGw, 1)).transpose(0,1).view((1, 1, nGh, nGw)).float()
    #grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)


def build_targets_max(target, anchor_wh, nA, nC, nGh, nGw):
    """
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)  # number of images in batch

    txy = torch.zeros(nB, nA, nGh, nGw, 2).cuda()  # batch size, anchors, grid size
    twh = torch.zeros(nB, nA, nGh, nGw, 2).cuda()
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tcls = torch.ByteTensor(nB, nA, nGh, nGw, nC).fill_(0).cuda()  # nC = number of classes
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda()
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long().cuda()
        t = t[:, [0, 2, 3, 4, 5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        # gxy, gwh = t[:, 1:3] * nG, t[:, 3:5] * nG
        gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gi = torch.clamp(gxy[:, 0], min=0, max=nGw - 1).long()
        gj = torch.clamp(gxy[:, 1], min=0, max=nGh - 1).long()

        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        # gi, gj = torch.clamp(gxy.long(), min=0, max=nG - 1).t()
        # gi, gj = gxy.long().t()

        # iou of targets-anchors (using wh only)
        box1 = gwh
        box2 = anchor_wh.unsqueeze(1)
        inter_area = torch.min(box1, box2).prod(2)
        iou = inter_area / (box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

        # Select best iou_pred and anchor
        iou_best, a = iou.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1:
            _, iou_order = torch.sort(-iou_best)  # best to worst

            # Unique anchor selection
            u = torch.stack((gi, gj, a), 0)[:, iou_order]
            # _, first_unique = np.unique(u, axis=1, return_index=True)  # first unique indices
            first_unique = return_torch_unique_index(u, torch.unique(u, dim=1))  # torch alternative
            i = iou_order[first_unique]
            # best anchor must share significant commonality (iou) with target
            i = i[iou_best[i] > 0.60]  # TODO: examine arbitrary threshold
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            t_id = t_id[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_best < 0.60:
                continue

        tc, gxy, gwh = t[:, 0].long(), t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh

        # XY coordinates
        txy[b, a, gj, gi] = gxy - gxy.floor()

        # Width and height
        twh[b, a, gj, gi] = torch.log(gwh / anchor_wh[a])  # yolo method
        # twh[b, a, gj, gi] = torch.sqrt(gwh / anchor_wh[a]) / 2 # power method

        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1
        tid[b, a, gj, gi] = t_id.unsqueeze(1)
    tbox = torch.cat([txy, twh], -1)
    return tconf, tbox, tid


def build_targets_thres(target, anchor_wh, nA, nC, nGh, nGw):
    ID_THRESH = 0.5
    FG_THRESH = 0.5
    BG_THRESH = 0.4
    nB = len(target)  # number of images in batch
    assert (len(anchor_wh) == nA)

    tbox = torch.zeros(nB, nA, nGh, nGw, 4).cuda()  # batch size, anchors, grid size
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda()
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long().cuda()
        t = t[:, [0, 2, 3, 4, 5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gxy[:, 0] = torch.clamp(gxy[:, 0], min=0, max=nGw - 1)
        gxy[:, 1] = torch.clamp(gxy[:, 1], min=0, max=nGh - 1)

        gt_boxes = torch.cat([gxy, gwh], dim=1)  # Shape Ngx4 (xc, yc, w, h)

        anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)
        anchor_list = anchor_mesh.permute(0, 2, 3, 1).contiguous().view(-1, 4)  # Shpae (nA x nGh x nGw) x 4
        # print(anchor_list.shape, gt_boxes.shape)
        iou_pdist = bbox_iou(anchor_list, gt_boxes)  # Shape (nA x nGh x nGw) x Ng
        iou_max, max_gt_index = torch.max(iou_pdist, dim=1)  # Shape (nA x nGh x nGw), both

        iou_map = iou_max.view(nA, nGh, nGw)
        gt_index_map = max_gt_index.view(nA, nGh, nGw)

        # nms_map = pooling_nms(iou_map, 3)

        id_index = iou_map > ID_THRESH
        fg_index = iou_map > FG_THRESH
        bg_index = iou_map < BG_THRESH
        ign_index = (iou_map < FG_THRESH) * (iou_map > BG_THRESH)
        tconf[b][fg_index] = 1
        tconf[b][bg_index] = 0
        tconf[b][ign_index] = -1

        gt_index = gt_index_map[fg_index]
        gt_box_list = gt_boxes[gt_index]
        gt_id_list = t_id[gt_index_map[id_index]]
        # print(gt_index.shape, gt_index_map[id_index].shape, gt_boxes.shape)
        if torch.sum(fg_index) > 0:
            tid[b][id_index] = gt_id_list.unsqueeze(1)
            fg_anchor_list = anchor_list.view(nA, nGh, nGw, 4)[fg_index]
            delta_target = encode_delta(gt_box_list, fg_anchor_list)
            tbox[b][fg_index] = delta_target
    return tconf, tbox, tid

def decode_delta_map(delta_map, anchors):
    '''
    :param: delta_map, shape (nB, nA, nGh, nGw, 4)
    :param: anchors, shape (nA,4)
    '''
    nB, nA, nGh, nGw, _ = delta_map.shape
    anchor_mesh = generate_anchor(nGh, nGw, anchors)
    anchor_mesh = anchor_mesh.permute(0,2,3,1).contiguous()              # Shpae (nA x nGh x nGw) x 4
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(nB,1,1,1,1)
    pred_list = decode_delta(delta_map.view(-1,4), anchor_mesh.view(-1,4))
    pred_map = pred_list.view(nB, nA, nGh, nGw, 4)
    return pred_map

def return_torch_unique_index(u, uv):
    n = uv.shape[1]  # number of columns
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:j + 1] == u).all(0).nonzero()[0]

    return first_unique

def generate_anchor(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx =torch.meshgrid(torch.arange(nGh), torch.arange(nGw))
    xx, yy = xx.cuda(), yy.cuda()

    mesh = torch.stack([xx, yy], dim=0)                                              # Shape 2, nGh, nGw
    mesh = mesh.unsqueeze(0).repeat(nA,1,1,1).float()                                # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nGh,nGw) # Shape nA x 2 x nGh x nGw
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)                       # Shape nA x 4 x nGh x nGw
    return anchor_mesh

def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw/pw)
    dh = torch.log(gh/ph)
    return torch.stack([dx, dy, dw, dh], dim=1)

def decode_delta(delta, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack([gx, gy, gw, gh], dim=1)

def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)
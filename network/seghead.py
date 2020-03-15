import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from network.backbone import ResNet
from roi_align import RoIAlign
from roi_align import CropAndResize

class SegHead(nn.Module):

    def __init__(self,img_size,device):
        super(SegHead, self).__init__()
        self.img_height=img_size[1]
        self.img_width=img_size[0]
        self.feature_height=math.ceil(self.img_height/16)
        self.feature_width = math.ceil(self.img_width / 16)
        self.device = device

        crop_height = 28
        crop_width = 14
        self.roi_align = RoIAlign(crop_height, crop_width, 0.25)

        self.box_index = torch.tensor([0], dtype=torch.int).cuda()

        self.seg = nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 256, (2, 2), stride=2, padding=2),
            torch.nn.ConvTranspose2d(256, 64, (2, 2), stride=2, padding=2),
            torch.nn.ConvTranspose2d(64, 16, (2, 2), stride=2, padding=2),
            torch.nn.Conv2d(16, 1, (1, 1))
        )

        self.zoomboxes = torch.Tensor([[0, 0, 84, 196]]).cuda()



    def forward(self,featuremap,bbox_list):
        x=[]
        for bbox in bbox_list:
            bbox = bbox.squeeze()
            boxes = self.format_box(bbox).cuda(self.device)
            crops = self.roi_align(featuremap, boxes, self.box_index)  # 输入必须是tensor，不能是numpy
            crops=self.seg(crops)
            zoom_roi_align = RoIAlign(bbox[3], bbox[2], 0.25)
            crops = zoom_roi_align(crops, self.zoomboxes, self.box_index)

            x.append(crops.squeeze())
        return x



    def format_box(self,bbox):
        return torch.Tensor([[bbox[0] / self.img_height * self.feature_height,
                              bbox[1] / self.img_width * self.feature_width,
                              (bbox[2]+bbox[0]) / self.img_height * self.feature_height,
                              (bbox[3]+bbox[1]) / self.img_width * self.feature_width]])

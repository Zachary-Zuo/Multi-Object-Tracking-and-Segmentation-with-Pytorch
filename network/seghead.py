import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from network.backbone import ResNet
from roi_align import RoIAlign
from roi_align import CropAndResize

class SegHead(nn.Module):

    def __init__(self,img_size):
        super(SegHead, self).__init__()
        self.img_height=img_size[1]
        self.img_width=img_size[0]
        self.feature_height=math.ceil(self.img_height/16)
        self.feature_width = math.ceil(self.img_width / 16)

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



    def forward(self,fms,bbox_list):
        x=[]
        for bbox in bbox_list:
            out = []
            for fm in fms:
                fw,fh = fm.shape[-2:]
                bbox = bbox.squeeze()
                boxes = self.format_box(bbox,fw,fh).cuda()
                crops = self.roi_align(fm, boxes, self.box_index)  # 输入必须是tensor，不能是numpy
                out.append(crops)
            output = torch.cat(out,dim=1)
            crops=self.seg(output)
            zoom_roi_align = RoIAlign(bbox[3], bbox[2], 0.25)
            crops = zoom_roi_align(crops, self.zoomboxes, self.box_index)

            x.append(crops.squeeze())
        return x



    def format_box(self,bbox,fw,fh):
        return torch.Tensor([[bbox[0] / self.img_height * fh,
                              bbox[1] / self.img_width * fw,
                              (bbox[2]+bbox[0]) / self.img_height * fh,
                              (bbox[3]+bbox[1]) / self.img_width * fw]])

if __name__=='__main__':
    net = SegHead((1920,1080))
    for ml1 in net.modules():
        if isinstance(ml1, nn.Sequential):
            for ml2 in ml1:
                if isinstance(ml2, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(ml2.weight.data)
                elif isinstance(ml2, nn.Conv2d):
                    nn.init.kaiming_normal_(ml2.weight.data)
        elif isinstance(ml1, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(ml1.weight.data)
        elif isinstance(ml1, nn.Conv2d):
            nn.init.kaiming_normal_(ml1.weight.data)
        else:
            print(type(ml1))
    torch.save(net.state_dict(), 'seg.pth')


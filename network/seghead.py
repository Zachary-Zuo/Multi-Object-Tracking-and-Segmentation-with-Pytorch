import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from network.backbone import ResNet
from roi_align import RoIAlign
from roi_align import CropAndResize

def make_conv3x3(
    in_channels,
    out_channels,
    dilation=1,
    stride=1,
    use_gn=False,
    kaiming_init=True
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False if use_gn else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)

    return conv

class SegHead(nn.Module):

    def __init__(self,img_size):
        super(SegHead, self).__init__()
        self.img_height=img_size[1]
        self.img_width=img_size[0]
        self.feature_height=math.ceil(self.img_height/16)
        self.feature_width = math.ceil(self.img_width / 16)

        self.crop_height = 28
        self.crop_width = 14

        self.box_index = torch.tensor([0], dtype=torch.int).cuda()

        self.blocks = []
        for id in range(4):
            layer_name = "mask_fcn{}".format(id)
            module = make_conv3x3(256, 256,dilation=1, stride=1)
            self.add_module(layer_name, module)
            self.blocks.append(layer_name)

        self.zoomboxes = torch.Tensor([[0, 0, 28, 56]]).cuda()

        self.conv5_mask = torch.nn.ConvTranspose2d(256, 256, 2, 2, 0)
        self.mask_fcn_logits = torch.nn.Conv2d(256, 1, 1, 1, 0)

        nn.init.kaiming_normal_(
            self.conv5_mask.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.conv5_mask.bias, 0)

        nn.init.kaiming_normal_(
            self.mask_fcn_logits.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.mask_fcn_logits.bias, 0)

        # for name, param in self.named_parameters():
        #     if "bias" in name:
        #         nn.init.constant_(param, 0)
        #     elif "weight" in name:
        #         nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def pooler(self,fms,bbox):
        sampling_ratio = 0.03125
        for i, fm in enumerate(reversed(fms)):
            fw, fh = fm.shape[-2:]
            if bbox[0]/self.img_height*fh > 6 and bbox[1]/self.img_width*fw>3:
                roi_align = RoIAlign(self.crop_height, self.crop_width, sampling_ratio)
                boxes = self.format_box(bbox, fw, fh).cuda()
                crops = roi_align(fm, boxes, self.box_index)  # 输入必须是tensor，不能是numpy
                return crops
            sampling_ratio *= 2
        fm = fms[0]
        fw, fh = fm.shape[-2:]
        roi_align = RoIAlign(self.crop_height, self.crop_width, sampling_ratio/2)
        boxes = self.format_box(bbox, fw, fh).cuda()
        crops = roi_align(fm, boxes, self.box_index)  # 输入必须是tensor，不能是numpy
        return crops


    def forward(self,fms,bbox_list):
        x=[]
        for bbox in bbox_list:
            bbox = bbox.squeeze()
            out = self.pooler(fms,bbox)
            for layer_name in self.blocks:
                out = F.relu(getattr(self, layer_name)(out))
            out = self.conv5_mask(out)
            out = self.mask_fcn_logits(out)
            zoom_roi_align = RoIAlign(bbox[3], bbox[2], 0.25)
            out = zoom_roi_align(out, self.zoomboxes, self.box_index)
            x.append(out.squeeze())
        return x



    def format_box(self,bbox,fw,fh):
        return torch.Tensor([[bbox[0] / self.img_height * fh,
                              bbox[1] / self.img_width * fw,
                              (bbox[2]+bbox[0]) / self.img_height * fh,
                              (bbox[3]+bbox[1]) / self.img_width * fw]])

if __name__=='__main__':
    net = SegHead((1920,1080))
    # for ml1 in net.modules():
    #     if isinstance(ml1, nn.Sequential):
    #         for ml2 in ml1:
    #             if isinstance(ml2, nn.ConvTranspose2d):
    #                 nn.init.kaiming_normal_(ml2.weight.data,mode="fan_out", nonlinearity="relu")
    #             elif isinstance(ml2, nn.Conv2d):
    #                 nn.init.kaiming_normal_(ml2.weight.data)
    #     elif isinstance(ml1, nn.ConvTranspose2d):
    #         nn.init.kaiming_normal_(ml1.weight.data)
    #     elif isinstance(ml1, nn.Conv2d):
    #         nn.init.kaiming_normal_(ml1.weight.data)
    #     else:
    #         print(type(ml1))
    torch.save(net.state_dict(), 'seg.pth')
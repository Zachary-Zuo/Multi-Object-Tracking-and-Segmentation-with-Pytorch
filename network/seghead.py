import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from network.backbone import ResNet
from roi_align import RoIAlign
import math
import torchvision
from roi_align import CropAndResize
import matplotlib.pyplot as plt

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
        self.s_mask = nn.Parameter(-9.9 * torch.ones(1))

        self.crop_height = 28
        # self.crop_width = 14
        self.crop_width = 28

        self.box_index = torch.tensor([0], dtype=torch.int).cuda()

        self.blocks = []
        for id in range(1,5):
            layer_name = "mask_fcn{}".format(id)
            module = make_conv3x3(256, 256,dilation=1, stride=1)
            self.add_module(layer_name, module)
            self.blocks.append(layer_name)

        # self.zoomboxes = torch.Tensor([[0, 0, 28, 56]]).cuda()
        self.zoomboxes = torch.Tensor([[0, 0, 56, 56]]).cuda()

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

    def pooler(self,fms,bbox=None,level=2):
        if bbox==None:
            return fms[level]
        else:
            a = math.sqrt(bbox[2]*bbox[3])/math.sqrt(self.img_width*self.img_height)
            if a==0:
                a+=0.01
            roi_level = min(2,max(0,5+math.log2(a)))
            roi_level = int(roi_level)
            # print(roi_level,bbox[2]*bbox[3])
            fm = fms[roi_level]
            fh, fw = fm.shape[-2:]
            sampling_ratio = 0.03125/(2**roi_level)
            if not self.training:
                # zoom_roi_align = RoIAlign(bbox[3], bbox[2], 0.25)
                # out = zoom_roi_align(out, self.zoomboxes, self.box_index)
                boxes = self.format_ops_box(bbox, fw, fh).cuda()
                crops = torchvision.ops.roi_align(fm,boxes,(self.crop_height, self.crop_width))[0].unsqueeze(0)
                return crops
            else:
                roi_align = RoIAlign(self.crop_height, self.crop_width, sampling_ratio)
                boxes = self.format_box(bbox, fw, fh).cuda()
                # print(fm.shape,boxes,bbox)
                crops = roi_align(fm, boxes, self.box_index)  # 输入必须是tensor，不能是numpy
                # crops = torchvision.ops.roi_align(fm,boxes,(28,28))[0].unsqueeze(0)
                return crops


    def forward(self,fms,bbox_list=None,gts=None,level=2):
        if bbox_list ==None:
            loss = []
            x = []
            for l in range(3):
                out = self.pooler(fms,level=l)
                for layer_name in self.blocks:
                    out = F.relu(getattr(self, layer_name)(out))
                out = self.conv5_mask(out)
                out = self.mask_fcn_logits(out)
                # plt.imshow(out[0][0].detach().cpu().numpy())
                # plt.show()
                if self.training:
                    maskloss = F.binary_cross_entropy_with_logits(out, gts[l])
                    loss.append(maskloss)
                else:
                    x.append(out.squeeze())
            if self.training:
                maskloss = sum(loss)
                return maskloss * torch.exp(-self.s_mask) * 0.5 + self.s_mask * 0.5, maskloss.item()
            else:
                return x

            # return out

        else:
            x=[]
            for bbox in bbox_list:
                bbox = bbox.squeeze()
                assert bbox[0]>=0 and bbox[1]>=0 and bbox[0]<bbox[2] and bbox[1]<bbox[3]
                out = self.pooler(fms,bbox)
                for layer_name in self.blocks:
                    out = F.relu(getattr(self, layer_name)(out))
                out = self.conv5_mask(out)
                out = self.mask_fcn_logits(out)
                if not self.training:
                    zoom_roi_align = RoIAlign(bbox[3], bbox[2], 0.25)
                    out = zoom_roi_align(out, self.zoomboxes, self.box_index)
                    # zoom = torch.Tensor([0, 0, 0, 56, 56])
                    # out = torchvision.ops.roi_align(out,zoom,(bbox[3], bbox[2]))[0]
                    # plt.imshow(out.squeeze().detach().cpu().numpy())
                    # plt.show()
                # plt.imshow(out.detach().cpu().numpy())
                # plt.show()
                x.append(out.squeeze())
            if self.training:
                out = torch.cat(x)
                maskloss = F.binary_cross_entropy_with_logits(out, gts)
                return  maskloss* torch.exp(-self.s_mask)*0.5 +self.s_mask*0.5,maskloss.item()
            return x



    def format_box(self,bbox,fw,fh):
        return torch.Tensor([[bbox[0] / self.img_height * fh,
                              bbox[1] / self.img_width * fw,
                              (bbox[2]+bbox[0]) / self.img_height * fh,
                              (bbox[3]+bbox[1]) / self.img_width * fw]])

    def format_ops_box(self,bbox,fw,fh):
        return torch.Tensor([0,bbox[0] / self.img_height * fh,
                              bbox[1] / self.img_width * fw,
                              (bbox[2]+bbox[0]) / self.img_height * fh,
                              (bbox[3]+bbox[1]) / self.img_width * fw])

def initialize_net(net):
    pretrained_dict = torch.load('seghead_epoch-999.pth',map_location='cuda:0')
    model_dict = net.state_dict()
    # pretrained_dict = pretrained_dict['model']
    pret = {}
    for mk in model_dict.keys():
        pret[mk]=model_dict[mk]
        for k,v in pretrained_dict.items():
            if mk in k:
                pret[mk]=v
                continue


    print(pret.keys())
    model_dict.update(pret)
    net.load_state_dict(model_dict)

if __name__=='__main__':
    # x = torch.randn(3, 256, 30, 62).cuda()
    net = SegHead((2048,1024))
    initialize_net(net)

    torch.save(net.state_dict(), 'seghead_epoch-999.pth')
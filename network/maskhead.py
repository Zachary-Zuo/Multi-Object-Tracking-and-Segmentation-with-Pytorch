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

class MaskHead(nn.Module):

    def __init__(self):
        super(MaskHead, self).__init__()
        self.crop_height = 28
        # self.crop_width = 14
        self.crop_width = 28

        self.box_index = torch.tensor([0], dtype=torch.int).cuda()
        self.s_mask = nn.Parameter(-9.9 * torch.ones(1))

        self.blocks = []
        for id in range(1, 5):
            layer_name = "mask_fcn{}".format(id)
            module = make_conv3x3(256, 256, dilation=1, stride=1)
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

        self.catconv1 = torch.nn.ConvTranspose2d(256, 256, 2, 2, 0)
        self.catconv2 = torch.nn.ConvTranspose2d(256, 256, 2, 2, 0)
        self.catconv3 = torch.nn.ConvTranspose2d(256, 256, 2, 2, 0)
        self.catconv4 = make_conv3x3(768, 256, dilation=1, stride=1)
        nn.init.kaiming_normal_(
            self.catconv1.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.catconv1.bias, 0)
        nn.init.kaiming_normal_(
            self.catconv2.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.catconv2.bias, 0)
        nn.init.kaiming_normal_(
            self.catconv3.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.catconv3.bias, 0)


    def pooler(self,fms,bbox=None,level=2):
        if level==0:
            return self.catconv1(self.catconv2(fms[level]))
        elif level==1:
            return self.catconv3(fms[level])
        else:
            return fms[level]

    def forward(self,fms,bbox_list=None,gts=None,level=2):
        loss = []
        x = []
        out = []
        for l in range(3):
            out.append(self.pooler(fms,level=l))
        out = torch.cat(out,1)
        out = self.catconv4(out)
        for layer_name in self.blocks:
            out = F.relu(getattr(self, layer_name)(out))
        out = self.conv5_mask(out)
        out = self.mask_fcn_logits(out)
        if self.training:
            maskloss = F.binary_cross_entropy_with_logits(out, gts)
            return maskloss * torch.exp(-self.s_mask) * 0.5 + self.s_mask * 0.5, maskloss.item()
        else:
            return out

def initialize_net(net):
    pretrained_dict = torch.load('seghead_epoch-999.pth',map_location='cuda:0')
    model_dict = net.state_dict()
    # pretrained_dict = pretrained_dict['model']
    pret = {}
    for mk in model_dict.keys():
        pret[mk]=model_dict[mk]
        if "s_mask" in mk:
            pret[mk]=torch.Tensor([-0.32])
            continue
        for k,v in pretrained_dict.items():
            if mk in k:
                pret[mk]=v
                continue


    print(pret.items())
    model_dict.update(pret)
    net.load_state_dict(model_dict)

if __name__=='__main__':
    # x = torch.randn(3, 256, 30, 62).cuda()
    x = []
    x.append(torch.randn(1, 256, 19, 34).cuda())
    x.append(torch.randn(1, 256, 38, 68).cuda())
    x.append(torch.randn(1, 256, 76, 136).cuda())
    net = MaskHead().cuda().eval()
    initialize_net(net)
    #
    torch.save(net.state_dict(), 'MaskHead_epoch-999.pth')
    # out = net(x)
    # print(out.shape)
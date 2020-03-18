import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from collections import OrderedDict



class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class StemWithFixedBatchNorm(nn.Module):
    def __init__(self):
        super(StemWithFixedBatchNorm, self).__init__()

        out_channels = 64

        self.conv1 = nn.Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = FrozenBatchNorm2d(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

class BottleneckWithFixedBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels=64,
        bottleneck_channels=64,
        out_channels=256,
        stride=1
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                FrozenBatchNorm2d(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        stride_1x1, stride_3x3 = 1,1

        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = FrozenBatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1,
            bias=False,
            dilation=1
        )
        nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = FrozenBatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = FrozenBatchNorm2d(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)

        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.stem = StemWithFixedBatchNorm()
        self.stages = []
        self.return_features = {}
        name = "layer" + str(1)
        module = nn.Sequential(
            BottleneckWithFixedBatchNorm(in_channels=64,bottleneck_channels=64,out_channels=256),
            BottleneckWithFixedBatchNorm(in_channels=256, bottleneck_channels=64, out_channels=256),
            BottleneckWithFixedBatchNorm(in_channels=256, bottleneck_channels=64, out_channels=256)
        )
        self.add_module(name, module)
        self.stages.append(name)
        self.return_features[name] = "layer1.2"


        name = "layer" + str(2)
        module = nn.Sequential(
            BottleneckWithFixedBatchNorm(in_channels=256, bottleneck_channels=128, out_channels=512,stride=2),
            BottleneckWithFixedBatchNorm(in_channels=512, bottleneck_channels=128, out_channels=512),
            BottleneckWithFixedBatchNorm(in_channels=512, bottleneck_channels=128, out_channels=512),
            BottleneckWithFixedBatchNorm(in_channels=512, bottleneck_channels=128, out_channels=512)
        )
        self.add_module(name, module)
        self.stages.append(name)
        self.return_features[name] = "layer2.3"


        name = "layer" + str(3)
        module = nn.Sequential(
            BottleneckWithFixedBatchNorm(in_channels=512, bottleneck_channels=256, out_channels=1024,stride=2),
            BottleneckWithFixedBatchNorm(in_channels=1024, bottleneck_channels=256, out_channels=1024),
            BottleneckWithFixedBatchNorm(in_channels=1024, bottleneck_channels=256, out_channels=1024),
            BottleneckWithFixedBatchNorm(in_channels=1024, bottleneck_channels=256, out_channels=1024),
            BottleneckWithFixedBatchNorm(in_channels=1024, bottleneck_channels=256, out_channels=1024),
            BottleneckWithFixedBatchNorm(in_channels=1024, bottleneck_channels=256, out_channels=1024)
        )
        self.add_module(name, module)
        self.stages.append(name)
        self.return_features[name] = "layer3.5"

        name = "layer" + str(4)
        module = nn.Sequential(
            BottleneckWithFixedBatchNorm(in_channels=1024, bottleneck_channels=512, out_channels=2048,stride=2),
            BottleneckWithFixedBatchNorm(in_channels=2048, bottleneck_channels=512, out_channels=2048),
            BottleneckWithFixedBatchNorm(in_channels=2048, bottleneck_channels=512, out_channels=2048)
        )
        self.add_module(name, module)
        self.stages.append(name)
        self.return_features[name] = "layer4.2"

        # self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs

class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]

class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """
    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)
            print("yes")

        return tuple(results)



class GeneralizedRCNN(nn.Module):
    def __init__(self):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = nn.Sequential(OrderedDict([
                  ('body', ResNet()),
                  ('fpn', FPN([256,512,1024,2048],256,nn.Conv2d,LastLevelMaxPool)),
                ]))

    def forward(self, x):
        return self.backbone(x)

def initialize_net(net):
    pretrained_dict = torch.load('model_final.pth')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'backbone' in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

if __name__=='__main__':
    x = torch.randn(1,3,1080,1920).cuda()
    net = GeneralizedRCNN().cuda()
    out = net(x)
    for o  in out:
        print(o.shape)
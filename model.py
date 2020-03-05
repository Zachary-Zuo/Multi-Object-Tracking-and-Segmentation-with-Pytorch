
import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import cv2
import matplotlib.pyplot as plt
import torchvision
from roi_align import RoIAlign
from roi_align import CropAndResize
from network.backbone import ResNet





if __name__=='__main__':
    device = torch.device("cuda:"+str(0) if torch.cuda.is_available() else "cpu")
    resnet = ResNet()
    C1, C2, C3, C4 = resnet.stages()
    img = cv2.imread('000001.jpg')
    print(img.shape)
    img = img[:800,:1078,:].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    img = img[None]
    print(img.shape)
    torchimg = torch.from_numpy(img)

    resnet = resnet.cuda()
    torchimg=torchimg.cuda()
    out = C1(torchimg)
    print(out.shape)
    out = C2(out)
    print(out.shape)
    out = C3(out)
    print(out.shape)
    out = C4(out)
    print("out:",out.shape)
    data=out.cpu().detach()

    # for example, we have two bboxes with coords xyxy (first with batch_id=0, second with batch_id=1).
    boxes = torch.Tensor([[10/1080*34, 20/1920*60, 540/1080*34, 640/1920*60]]).cuda()
    #做好坐标比例变化


    box_index = torch.tensor([0, 1], dtype=torch.int).cuda()  # index of bbox in batch

    # RoIAlign layer with crop sizes:
    crop_height = 28
    crop_width = 14
    roi_align = RoIAlign(crop_height, crop_width,0.25)

    # make crops:
    crops = roi_align(out.cuda(), boxes, box_index) #输入必须是tensor，不能是numpy
    print(type(crops))
    net = torch.nn.ConvTranspose2d(1024, 256, (2, 2), stride=2, padding=2).cuda()
    net1 = torch.nn.ConvTranspose2d(256, 64, (2, 2), stride=2, padding=2).cuda()
    net2 = torch.nn.ConvTranspose2d(64, 16, (2, 2), stride=2, padding=2).cuda()
    net3 = torch.nn.Conv2d(16, 1, (1, 1)).cuda()
    crops = net(crops)
    crops = net1(crops)
    crops = net2(crops)
    crops = net3(crops)

    # plt.imshow(img[0][0])
    # plt.show()
    print("crops",crops.shape)
    crops=crops.cpu().detach().numpy()
    plt.imshow(crops[0][0])
    plt.show()

    # ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2)

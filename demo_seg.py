from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
from datetime import datetime
import socket
import timeit
from tensorboardX import SummaryWriter
import numpy as np
import pycocotools.mask as rletools
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

import torch.nn.functional as F
from network.backbone import ResNet
from network.seghead import SegHead
import dataloaders.MOTS_dataloaders as ms
import dataloaders.correct_dataloader as cor
from network.fpn import FPN101

def get_img_size(sequence):
    if sequence==5 or sequence==6:
        return [640,480]
    else:
        return [1920,1080]

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def main(sequence):
    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    backbone = FPN101()
    seghead=SegHead(get_img_size(sequence))
    BackBoneName = "FPN101"
    SegHeadName = "seghead"

    backbone.load_state_dict(
            torch.load(os.path.join(save_dir, BackBoneName + '_epoch-' + str(50) + '.pth'),
                       map_location=lambda storage, loc: storage))
    seghead.load_state_dict(
        torch.load(os.path.join(save_dir, SegHeadName + '_epoch-' + str(50) + '.pth'),
                   map_location=lambda storage, loc: storage))

    backbone=backbone.cuda()
    seghead=seghead.cuda()

    test = cor.MOTSCorrectDataset(sequence=sequence)

    testloader = DataLoader(test)


    backbone.eval()
    seghead.eval()
    file = open('{:04}.txt'.format(sequence), "w")

    with torch.no_grad():


        for ii, sample_batched in enumerate(testloader):
            inputs, bbox,track_list = sample_batched["img"],sample_batched["bbox"],sample_batched["track"]

            inputs = inputs.cuda()
            feature = backbone(inputs)

            out = seghead(feature,bbox)

            background = np.zeros_like(inputs[0][0].cpu())
            background = background.astype(np.uint8)
            result_list=[]

            for pre,nbox,track_id in zip(out,bbox,track_list):
                box = nbox.squeeze()
                box = [int(i) for i in box]

                pre = pre.cpu().detach().numpy()
                pre = normalization(pre)
                result_list.append((pre,box,track_id))
            result_list.sort(key=lambda item:(item[0]>0.5).sum()/(item[1][3]*item[1][2]))


            for item in result_list:
                pre = item[0]
                box = item[1]
                track_id = item[2]

                mask = np.zeros_like(inputs[0][0].cpu())
                # box = box.squeeze()
                # box = [int(i) for i in box]

                temp = mask[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
                # pre = pre.cpu().detach().numpy()
                # pre=normalization(pre)
                temp[pre>0.5] = 1
                background[mask>0]=int(track_id)
            # print(background[500:600,500:600])
            # print(track_id)

            for id in track_id:
                id = int(id)
                mask = np.zeros_like(inputs[0][0].cpu())
                # print(id)
                mask[background==id]=1
                mask = np.asfortranarray(mask)
                mask = mask.astype(np.uint8)
                rle = rletools.encode(mask)
                output =' '.join([str(ii+1),str(int(2000+track_id)),"2",str(rle['size'][0]),str(rle['size'][1]),rle['counts'].decode(encoding='UTF-8')])
                file.write(output+'\n')
    file.close()

if __name__ == "__main__":
    main(2)
    main(5)
    main(9)
    main(11)
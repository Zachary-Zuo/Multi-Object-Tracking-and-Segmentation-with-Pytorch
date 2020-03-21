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
import cv2
import torch.nn.functional as F
from network.backbone import ResNet
from network.seghead import SegHead

import dataloaders.MOTS_dataloaders as ms
import dataloaders.Demo_dataloader as de
from network.fpn import FPN101
from network.GeneralizedRCNN import GeneralizedRCNN

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

    backbone = GeneralizedRCNN()
    # seghead=SegHead(get_img_size(sequence))
    seghead = SegHead([2048,1024])
    BackBoneName = "GeneralizedRCNN"
    SegHeadName = "seghead"

    backbone.load_state_dict(
            torch.load(os.path.join(save_dir, BackBoneName + '_epoch-' + str(32) + '.pth'),
                       map_location=lambda storage, loc: storage))

    seghead.load_state_dict(
        torch.load(os.path.join(save_dir, SegHeadName + '_epoch-' + str(32) + '.pth'),
                   map_location=lambda storage, loc: storage))

    backbone=backbone.cuda()
    seghead=seghead.cuda()

    test = de.MOTTrackDataset(sequence=sequence)

    testloader = DataLoader(test)


    backbone.eval()
    seghead.eval()
    file = open('{:04}.txt'.format(sequence), "w")

    size = get_img_size(sequence)
    with torch.no_grad():


        for ii, sample_batched in enumerate(testloader):
            inputs, bbox,track_list = sample_batched["img"],sample_batched["bbox"],sample_batched["track"]

            inputs = inputs.cuda()
            feature = backbone(inputs)

            output_list = []
            for l in range(1):
                out = seghead(feature,level=3-l)
                output = out.detach().cpu().numpy()[0][0]
                # plt.imshow(output)
                # plt.show()
                if sequence == 5 or sequence == 6:
                    output = cv2.resize(output, (640, 480))
                else:
                    output = cv2.resize(output, (1920, 1080))
                output_list.append(output)
            output = sum(output_list)
            # output = normalization(output)
            output[output < 0.5] = 0
            # plt.imshow(output)
            # plt.show()

            result_list=[]
            for nbox,track_id in zip(bbox,track_list):
                box = nbox.squeeze()
                box = [box[0]/2048*size[0],box[1]/1024*size[1],box[2]/2048*size[0],box[3]/1024*size[1]]
                box = [int(i) for i in box]
                result_list.append((box,track_id))
            result_list.sort(key=lambda item:item[0][3]*item[0][2])
            result_list.reverse()


            for item in result_list:
                box = item[0]
                track_id = item[1]

                mask = np.zeros_like(output)
                temp = mask[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
                # pre=normalization(pre)
                temp[temp>-1] = 1

                real = output.copy()
                real[mask<1]=0
                output[mask>0]=0
                real[real>0]=1
                # plt.imshow(real)
                # plt.show()
                mask = np.asfortranarray(real)
                mask = mask.astype(np.uint8)
                rle = rletools.encode(mask)
                # print(rletools.area(rle))
                if rletools.area(rle) < 10:
                    continue
                line = ' '.join([str(ii + 1), str(int(2000 + track_id)), "2", str(rle['size'][0]), str(rle['size'][1]),
                                   rle['counts'].decode(encoding='UTF-8')])
                file.write(line + '\n')
            #
            #
            # if sequence==5 or sequence==6:
            #     background = cv2.resize(background, (640, 480))
            # else:
            #     background = cv2.resize(background, (1920, 1080))
            #
            #
            #
            # for id in track_list:
            #     id = int(id)
            #     mask = np.zeros_like(background)
            #     mask[background==id]=1
            #     mask = np.asfortranarray(mask)
            #     mask = mask.astype(np.uint8)
            #     rle = rletools.encode(mask)
            #     # print(id,rletools.area(rle))
            #     if rletools.area(rle)<2000:
            #         continue
            #     output =' '.join([str(ii+1),str(int(2000+id)),"2",str(rle['size'][0]),str(rle['size'][1]),rle['counts'].decode(encoding='UTF-8')])
            #     file.write(output+'\n')
    file.close()

if __name__ == "__main__":
    main(2)
    print("finish:2")
    main(5)
    print("finish:5")
    main(9)
    print("finish:9")
    main(11)
    print("finish:11")

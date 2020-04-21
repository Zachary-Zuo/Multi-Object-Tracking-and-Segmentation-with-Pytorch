import argparse
import os
from datetime import datetime
import socket
import timeit
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn

import torch.nn.functional as F
from network.backbone import ResNet
from network.maskhead import MaskHead
from network.fpn import FPN101
from dataloaders.MOTS_dataloaders import JointDataset
from network.Darknet import Darknet,parse_model_cfg
import cv2

def get_img_size(sequence):
    if sequence==5:
        return [640,480]
    else:
        return [1920,1080]

def main(cfg):
    gpu_id = cfg.gpu_id
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu_id)

    # # Setting other parameters
    resume_epoch = 0  # Default is 0, change if want to resume
    nEpochs = 1000  # Number of epochs for training (500.000/2079)
    batch_size = cfg.batch_size
    snapshot = 10  # Store a model every snapshot epochs
    beta = 0.001
    margin = 0.3

    lr_B = 0.0001
    lr_S = 0.0001
    wd = 0.00002

    save_root_dir = "models"
    # save_dir = os.path.join(save_root_dir,"{:04}".format(sequence))
    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = JointDataset(transform=data_transforms,level=3,batchsize=batch_size,img_size=(2176, 1216))
    trainloader = DataLoader(dataset,batch_size=batch_size)

    cfg_dict = parse_model_cfg('cfg/yolov3_1088x608.cfg')
    backbone = Darknet(cfg_dict,dataset.nID)
    seghead=MaskHead()
    BackBoneName = "Darknet"
    SegHeadName = "MaskHead"

    backbone.load_state_dict(
            torch.load(os.path.join(save_dir, BackBoneName + '_epoch-' + str(999) + '.pth'),
                       map_location=lambda storage, loc: storage))
    seghead.load_state_dict(
        torch.load(os.path.join(save_dir, SegHeadName + '_epoch-' + str(999) + '.pth'),
                   map_location=lambda storage, loc: storage))

    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir, comment='-parent')


    backbone=backbone.cuda(device)
    seghead=seghead.cuda(device)


    # Use the following optimizer
    optimizerB = optim.Adam(backbone.parameters(), lr=lr_B, weight_decay=wd)
    optimizerS = optim.Adam(seghead.parameters(), lr=lr_S, weight_decay=wd)

    ii=0
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        # for lev,trainloader in enumerate(loaders):
        for sample_batched in trainloader:
            ii+=1
            img,gts,targets,targetslen = sample_batched["img"],sample_batched["mask"],sample_batched["targets"],sample_batched["targetslen"]
            # plt.imshow(img[0][0])
            # plt.show()
            # plt.imshow(gts[0][0])
            # plt.show()
            img.requires_grad_()
            img = img.cuda(device)
            targets = targets.cuda()
            targetslen = targetslen.cuda()
            # print(gts)
            # gts=gts.squeeze()
            gts = gts.cuda(device)
            loss, components ,featuremap = backbone.forward(img,targets,targetslen)

            maskloss,masklossitem = seghead(featuremap,gts=gts)

            losses = loss + maskloss

            losses = torch.mean(losses)


            backbone.zero_grad()
            seghead.zero_grad()
            losses.backward()
            optimizerB.step()
            optimizerS.step()

            if ii % 1 == 0:
                print(
                    "Iters: [%2d] time: %4.4f, Yolo_loss: %.8f,Mask_loss: %.8f,losses: %.8f"
                    % (ii, timeit.default_timer() - start_time,loss.item(),masklossitem,losses.item())
                )
            if ii % 2 == 0:
                writer.add_scalar('data/Yolo_loss_iter', loss.item(), ii)
                writer.add_scalar('data/Mask_loss_iter', masklossitem, ii)
                writer.add_scalar('data/losses_iter', losses, ii)

            if ii % 100 == 0:
                torch.save(backbone.state_dict(),
                           os.path.join(save_dir, BackBoneName + '_epoch-' + str(epoch) + '.pth'))
                torch.save(seghead.state_dict(), os.path.join(save_dir, SegHeadName + '_epoch-' + str(epoch) + '.pth'))

        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time))
        print("save models")
        torch.save(backbone.state_dict(), os.path.join(save_dir, BackBoneName + '_epoch-' + str(epoch) + '.pth'))
        torch.save(seghead.state_dict(), os.path.join(save_dir, SegHeadName + '_epoch-' + str(epoch) + '.pth'))
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--gpu_id', type=int, default=0, help='tracking buffer')
    parser.add_argument('--batch_size', type=int, default=1, help='tracking buffer')
    opt = parser.parse_args()
    print(opt)
    main(opt)
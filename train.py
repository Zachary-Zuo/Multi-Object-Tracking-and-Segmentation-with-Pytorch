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
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn

import torch.nn.functional as F
from network.backbone import ResNet
from network.seghead import SegHead
from network.fpn import FPN101
import dataloaders.MOTS_dataloaders as ms

def get_img_size(sequence):
    if sequence==5:
        return [640,480]
    else:
        return [1920,1080]

def main(sequence,cfg):
    gpu_id = cfg.gpu_id
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu_id)

    # # Setting other parameters
    resume_epoch = 0  # Default is 0, change if want to resume
    nEpochs = 1000  # Number of epochs for training (500.000/2079)
    batch_size = 1
    snapshot = 10  # Store a model every snapshot epochs
    beta = 0.001
    margin = 0.3

    lr_B = 1e-5
    lr_S = 1e-5
    wd = 0.0002

    save_root_dir = "models"
    # save_dir = os.path.join(save_root_dir,"{:04}".format(sequence))
    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    backbone = FPN101()
    seghead=SegHead(get_img_size(sequence))
    BackBoneName = "FPN101"
    SegHeadName = "seghead"

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

    ms_train = ms.MOTSDataset(sequence=sequence)

    trainloader = DataLoader(ms_train, batch_size=batch_size,num_workers=2)  # change to 1.2.0

    num_img_tr = len(trainloader)

    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        for ii, sample_batched in enumerate(trainloader):

            inputs, bbox,gts = sample_batched["img"], sample_batched["bbox"],sample_batched["mask"]

            inputs.requires_grad_()
            inputs = inputs.cuda(device)
            feature = backbone.forward(inputs)

            out = seghead(feature,bbox)
            gts = [gt.squeeze().cuda(device) for gt in gts]

            losses = []
            for pre,gt in zip(out,gts):
                losses.append(F.binary_cross_entropy_with_logits(pre,gt))

            # for pre, gt in zip(out, gts):
            #     gt=gt.cpu().detach().numpy()
            #     pre = pre.cpu().detach().numpy()
            #     plt.imshow(pre)
            #     plt.show()
            #     plt.imshow(gt)
            #     plt.show()

            loss = sum(losses)
            backbone.zero_grad()
            seghead.zero_grad()
            loss.backward()
            optimizerB.step()
            optimizerS.step()

            if (ii + num_img_tr * epoch) % 5 == 0:
                print(
                    "Iters: [%2d] time: %4.4f, loss: %.8f"
                    % (ii + num_img_tr * epoch, timeit.default_timer() - start_time,loss.item())
                )
            if (ii + num_img_tr * epoch) % 10 == 0:
                writer.add_scalar('data/loss_iter', loss.item(), ii + num_img_tr * epoch)
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time))
        print("save models")
        if epoch%50==0:
            torch.save(backbone.state_dict(), os.path.join(save_dir, BackBoneName + '_epoch-' + str(epoch) + '.pth'))
            torch.save(seghead.state_dict(), os.path.join(save_dir, SegHeadName + '_epoch-' + str(epoch) + '.pth'))
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--gpu_id', type=int, default=0, help='tracking buffer')
    opt = parser.parse_args()
    for i in range(100):
        main(2,opt)
        main(5,opt)
        main(9,opt)
        main(11,opt)
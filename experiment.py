from tools.mots_tools.io import *
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



seqs_list_file=r'E:\Challenge\MOTSChallenge\train\instances_txt'
filename = os.path.join(seqs_list_file,"{:04}.txt".format(2))
instance = load_txt(filename)
obj = instance[1][0]
print(obj.key())
mask = rletools.decode(obj.mask)
print(type(mask))
print(mask.shape)
print(rletools.encode(mask))
print(' '.join(["1","2","3"]))

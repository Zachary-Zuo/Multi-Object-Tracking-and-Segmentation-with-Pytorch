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




A=torch.ones(2,3) #2x3的张量（矩阵）

for i in A:
    print(i)

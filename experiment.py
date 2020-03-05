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

print(math.ceil(11.5))
print(math.ceil(11.2))
print(math.ceil(11.3))

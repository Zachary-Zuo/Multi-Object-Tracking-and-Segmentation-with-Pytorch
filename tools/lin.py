import cv2
import os

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from mypredictor import COCODemo
import os
import cv2

import sys

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    #response = requests.get(url)
    #pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    pil_image = Image.open(url).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

if __name__=='__main__':
    fps = 16
    size = (1080, 608)

    videowriter = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    imgPath = r"E:\Challenge\Towards-Realtime-MOT-master\results\frame-0"

    imgDir = os.listdir(imgPath)


    for frame in range(len(imgDir)):
        img = os.path.join(imgPath, "{:05}.jpg".format(frame))
        image = cv2.imread(img)

        videowriter.write(image)
        if frame%100==0:
            print("Frame:{}".format(frame))
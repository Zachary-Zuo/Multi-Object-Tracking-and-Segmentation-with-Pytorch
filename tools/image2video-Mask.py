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

def output_video(instance):
    fps = 8
    size = (1242, 375)
    sys.path.append(r"E:\Challenge\MaskR-CNN")
    config_file = "./configs/my_test_e2e_mask_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.7,
    )


    videowriter = cv2.VideoWriter("predictions-{}.avi".format(instance), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    imgPath = os.path.join(r"E:\Challenge\KITTI_MOTS\training\image_02","{:04}".format(instance))

    imgDir = os.listdir(imgPath)


    for frame in range(len(imgDir)):
        img = os.path.join(imgPath, "{:06}.png".format(frame))
        image = load(img)

        # compute predictions
        predictions = coco_demo.run_on_opencv_image(image)
        videowriter.write(predictions)
        if frame%10==0:
            print("Instance:{}   Frame:{}".format(instance,frame))


if __name__=="__main__":
    for i in range(21):
        if i==9:
            continue
        output_video(i)

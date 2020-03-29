import argparse
import numpy as np
import torch
import random
import cv2
import math
import time
import matplotlib.pyplot as plt
import cv2
import torchvision
from roi_align import RoIAlign
from network.seghead import SegHead
from network.GeneralizedRCNN import GeneralizedRCNN


def track(opt):
    print(opt.cfg)

def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height)/shape[0], float(width)/shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

def get_data(img_path,arguement):
    height = 1024
    width = 2048
    img = cv2.imread(img_path)  # BGR
    plt.imshow(img)
    plt.show()
    if img is None:
        raise ValueError('File corrupt {}'.format(img_path))
    augment_hsv = True
    if arguement and augment_hsv:
        # SV augmentation by 50%
        fraction = 0.50
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        S = img_hsv[:, :, 1].astype(np.float32)
        V = img_hsv[:, :, 2].astype(np.float32)

        a = (random.random() * 2 - 1) * fraction + 1
        S *= a
        if a > 1:
            np.clip(S, a_min=0, a_max=255, out=S)

        a = (random.random() * 2 - 1) * fraction + 1
        V *= a
        if a > 1:
            np.clip(V, a_min=0, a_max=255, out=V)

        img_hsv[:, :, 1] = S.astype(np.uint8)
        img_hsv[:, :, 2] = V.astype(np.uint8)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

    h, w, _ = img.shape
    img, ratio, padw, padh = letterbox(img, height=height, width=width)

    # Augment image and labels
    if arguement:
        img = random_affine(img,degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))


    img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB
    # if self.transforms is not None:
    #     img = self.transforms(img)

    return img, img_path, (h, w)

def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

        return imw, targets, M
    else:
        return imw

def format_box(bbox,fw,fh):
    return torch.Tensor([[bbox[0] / 1024 * fh,
                            bbox[1] / 2048 * fw,
                         (bbox[2] + bbox[0]) / 1024 * fh,
                            (bbox[3]+bbox[1]) / 2048 * fw]])

if __name__ == '__main__':
    # img,a,b = get_data(r'E:\Challenge\MOTSChallenge\train\images\0002\000001.jpg',False)
    # img = cv2.imread(r'E:\Challenge\MOTSChallenge\train\images\0002\000001.jpg')
    # print(img.shape)
    # img=cv2.resize(img,(2048,1024))  #修改图片的尺寸
    # # img, _, _, _ = letterbox(img, height=1024, width=2048)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # x = torch.randn(3,3,1024,2048).cuda()
    # net = GeneralizedRCNN().cuda()
    # seg = SegHead([2048, 1024]).cuda()
    # backbone.load_state_dict(
    #     torch.load(os.path.join(save_dir, BackBoneName + '_epoch-' + str(999) + '.pth'),
    #                map_location=lambda storage, loc: storage))
    # seghead.load_state_dict(
    #     torch.load(os.path.join(save_dir, SegHeadName + '_epoch-' + str(999) + '.pth'),
    #                map_location=lambda storage, loc: storage))

    id = set()
    id.add(1)
    if 1 in id:
        print("yes")
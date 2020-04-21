import cv2
from torch.utils.data import Dataset
from tools.mots_tools.io import *
import torch
from roi_align import RoIAlign
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import math

def format_box(bbox):
    return torch.Tensor([[bbox[0], bbox[1], bbox[0]+ bbox[2], bbox[1] + bbox[3]]])

def pass_box(bbox):
    return torch.Tensor([bbox[0], bbox[1], bbox[2], bbox[3]])

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    # x, y are coordinates of center
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively.
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def letterbox(img,mask,height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height)/shape[0], float(width)/shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_AREA)
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0, 0))

    return img,mask, ratio, dw, dh

def random_affine(img,mask, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
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
    mask = cv2.warpPerspective(mask, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=(0, 0, 0))  # BGR order borderValue

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

        return imw,mask, targets, M
    else:
        return imw

class MOTSDataset(Dataset):
    def __init__(self, inputRes=None,
                 # seqs_list_file='/home/zuochenyu/datasets/MOTSChallenge/train/instances_txt',
                 seqs_list_file=r'E:\Challenge\MOTSChallenge\train\instances_txt',
                 transform=None,
                 sequence=2,
                 random_rev_thred=0.4):

        self.imgPath = os.path.join(r'E:\Challenge\MOTSChallenge\train\images',"{:04}".format(sequence))
        # self.imgPath = os.path.join('/home/zuochenyu/datasets/MOTSChallenge/train/images', "{:04}".format(sequence))
        filename = os.path.join(seqs_list_file, "{:04}.txt".format(sequence))
        self.instance = load_txt(filename)
        self.transform = transform
        self.inputRes = inputRes
        self.random_rev_thred = random_rev_thred

        self.roi_align = RoIAlign(56, 56, 0.25)

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, idx):
        frame=idx+1
        instance_per_frame = self.instance[frame]
        bbox_list = []
        mask_list = []
        img = os.path.join(self.imgPath, "{:06}.jpg".format(frame))
        img = cv2.imread(img)
        img = cv2.resize(img, (2048, 1024))
        img = img[:,:,:].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        for obj in instance_per_frame:
            if obj.class_id!=2:
                continue
            mask = rletools.decode(obj.mask)
            mask = cv2.resize(mask, (2048, 1024))
            newmask = np.asfortranarray(mask)
            newmask = newmask.astype(np.uint8)
            newmask = rletools.encode(newmask)
            mask = torch.from_numpy(mask)
            mask = mask.float()
            mask = mask[None]
            mask = mask[None]
            mask = mask.contiguous()
            boxes = format_box(rletools.toBbox(newmask))
            bbox = pass_box(rletools.toBbox(newmask))
            box_index = torch.tensor([0], dtype=torch.int)

            # crop_height = int(bbox[3])
            # crop_width = int(bbox[2])
            # roi_align = RoIAlign(crop_height, crop_width, 0.25)


            crops = self.roi_align(mask, boxes, box_index)
            # print(boxes)
            # crops = torchvision.ops.roi_align(mask, boxes, (56, 56))[0]
            crops=crops.squeeze()
            mask_list.append(crops)
            bbox_list.append(bbox)


        if self.transform is not None:
            img = self.transform(img)
        return  {"img":img,"mask":mask_list,"bbox":bbox_list}


class JointDataset(Dataset):  # for training
    def __init__(self,
                 seqs_list_file='/home/zuochenyu/datasets/MOTSChallenge/train/instances_txt',
                 # seqs_list_file=r'E:\Challenge\MOTSChallenge\train\instances_txt',
                 # img_file_root=r'E:\Challenge\MOTSChallenge\train\images',
                 img_file_root='/home/zuochenyu/datasets/MOTSChallenge/train/images',
                 img_size=(1088, 608),
                 augment=True,
                 transform=None,
                 batchsize=1,
                 random_rev_thred=0.4,level=3):

        self.transform = transform
        self.tr_image = transforms.Compose([transforms.ToTensor()])
        self.level = level
        self.nID = 0
        self.img_list = []

        for sequence in [2,5,9,11]:
            imgPath = os.path.join(img_file_root, "{:04}".format(sequence))
            filename = os.path.join(seqs_list_file, "{:04}.txt".format(sequence))
            instance = load_txt(filename)
            for i in range(len(instance)):
                frame = i+1
                self.img_list.append((os.path.join(imgPath, "{:06}.jpg".format(frame)),instance[frame],sequence))
        self.nID = 14455
        random.shuffle(self.img_list)
        self.roi_align = RoIAlign(56, 56, 0.25)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.lastsize=0
        self.batchsize=batchsize

    def __len__(self):
        return len(self.img_list*self.batchsize)

    def __getitem__(self, idx):
        frame = idx//self.batchsize
        instance_per_frame = self.img_list[frame][1]
        img_path = self.img_list[frame][0]
        base = 500 * self.img_list[frame][2]

        if idx%self.batchsize==0:
            imgs,mask, labels, img_path, (h, w) = self.get_data(img_path, instance_per_frame,base)
            self.lastsize=int(labels.shape[0])
            return {"img": imgs, "mask": mask, "targets": labels, "targetslen": torch.Tensor([int(labels.shape[0]),])}
        else:
            while(True):
                imgs, mask, labels, img_path, (h, w) = self.get_data(img_path, instance_per_frame, base)
                if self.lastsize == int(labels.shape[0]):
                    return {"img": imgs, "mask": mask, "targets": labels,
                        "targetslen": torch.Tensor([int(labels.shape[0]), ])}

    def get_data(self, img_path, instance,baseID):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = True
        if self.augment and augment_hsv:
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

        shape = img.shape[:2]
        background = np.zeros([shape[0], shape[1]])
        labels0 = []
        for obj in instance:
            if obj.class_id != 2:
                continue
            mask = rletools.decode(obj.mask)
            background[mask > 0] = 255
            box = rletools.toBbox(rletools.encode(mask))
            labels0.append([0, obj.track_id % 1000 + baseID, (box[0] + box[2] / 2) / shape[1], (box[1] + box[3] / 2) / shape[0],
                            box[2] / shape[1], box[3] / shape[0]])

        h, w, _ = img.shape
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(background)
        # plt.show()
        img, mask, ratio, padw, padh = letterbox(img,background, height=height, width=width)

        # Load labels
        labels0 = np.array(labels0,dtype=np.float32)
        labels = labels0.copy()
        labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
        labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
        labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
        labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh

        # Augment image and labels
        if self.augment:
            img,mask, labels, M = random_affine(img,mask, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                mask = np.fliplr(mask)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()
        mask[mask>0]=1

        mask = cv2.resize(mask, (int(width/4), int(height/4)))
        # plt.imshow(mask)
        # plt.show()

        # background_list = []
        #
        # for l in range(3):
        #     if l == 0:
        #         background_list.append(self.transform(cv2.resize(mask, (68, 38))))
        #     elif l == 1:
        #         background_list.append(self.transform(cv2.resize(mask, (136, 76))))
        #     elif l == 2:
        #         background_list.append(self.transform(cv2.resize(mask, (272, 152))))
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img,mask, labels, img_path, (h, w)



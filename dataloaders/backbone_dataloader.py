import cv2
from torch.utils.data import Dataset
from tools.mots_tools.io import *
import torch
from roi_align import RoIAlign
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import random

def format_box(bbox):
    return torch.Tensor([[bbox[0], bbox[1], bbox[0]+ bbox[2], bbox[1] + bbox[3]]])

def pass_box(bbox):
    return torch.Tensor([bbox[0], bbox[1], bbox[2], bbox[3]])

class BackboneDataset(Dataset):
    def __init__(self, inputRes=None,
                 # seqs_list_file='/home/zuochenyu/datasets/MOTSChallenge/train/instances_txt',
                 seqs_list_file=r'E:\Challenge\MOTSChallenge\train\instances_txt',
                 img_file_root=r'E:\Challenge\MOTSChallenge\train\images',
                 # img_file_root='/home/zuochenyu/datasets/MOTSChallenge/train/images',
                 transform=None,
                 random_rev_thred=0.4,level=3):

        # self.imgPath = os.path.join(, "{:04}".format(sequence))
        self.transform = transform
        self.inputRes = inputRes
        self.random_rev_thred = random_rev_thred
        self.tr_image = transforms.Compose([transforms.ToTensor()])
        self.level = level

        self.img_list = []

        for sequence in [2,5,9,11]:
            imgPath = os.path.join(img_file_root, "{:04}".format(sequence))
            filename = os.path.join(seqs_list_file, "{:04}.txt".format(sequence))
            instance = load_txt(filename)
            for i in range(len(instance)):
                frame = i+1
                self.img_list.append((os.path.join(imgPath, "{:06}.jpg".format(frame)),instance[frame]))
        random.shuffle(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        instance_per_frame = self.img_list[idx][1]
        img = self.img_list[idx][0]
        img = cv2.imread(img)
        img = cv2.resize(img, (2048, 1024))
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        # img = img[:,:,:].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img, dtype=np.float32)
        # img /= 255.0
        background = np.zeros([1024,2048])
        for obj in instance_per_frame:
            if obj.class_id!=2:
                continue
            mask = rletools.decode(obj.mask)
            mask = cv2.resize(mask, (2048, 1024))
            background[mask>0]=255
        if self.level==0:
            background = cv2.resize(background, (1020, 508))
        elif self.level==1:
            background = cv2.resize(background, (508, 252))
        elif self.level == 2:
            background = cv2.resize(background, (252, 124))
        elif self.level == 3:
            background = cv2.resize(background, (124, 60))
        background = Image.fromarray(background.astype('uint8'))

        if self.transform is not None:
            random.seed(seed)  # apply this seed to img tranfsorms
            img = self.transform(img)
            img = self.tr_image(img)
            random.seed(seed)  # apply this seed to img tranfsorms
            background = self.transform(background)
            background = self.tr_image(background)
        return  {"img":img,"mask":background}
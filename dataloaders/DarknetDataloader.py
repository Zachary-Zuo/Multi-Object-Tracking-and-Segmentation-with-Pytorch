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

class DarknetDataset(Dataset):
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
        self.width = 1088
        self.height = 608
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

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        instance_per_frame = self.img_list[idx][1]
        img = self.img_list[idx][0]
        base = 500*self.img_list[idx][2]
        img = cv2.imread(img)
        img = cv2.resize(img, (self.width, self.height))
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        # img = img[:,:,:].transpose(2, 0, 1)
        # img = np.ascontiguousarray(img, dtype=np.float32)
        # img /= 255.0
        background = np.zeros([self.height,self.width])
        targets = []
        mask_list = []
        bbox_list = []


        for obj in instance_per_frame:
            if obj.class_id!=2:
                continue
            mask = rletools.decode(obj.mask)
            mask = cv2.resize(mask, (self.width, self.height))
            background[mask>0]=255
            mask = Image.fromarray(mask.astype('uint8'))
            random.seed(seed)  # apply this seed to img tranfsorms
            mask = self.transform(mask)
            mask = np.array(mask)
            mask[mask>0]=1
            mask = np.asfortranarray(mask, dtype=np.uint8)
            box = rletools.toBbox(rletools.encode(mask))
            targets.append([0,obj.track_id%1000+base,(box[0]+box[2]/2)/1088,(box[1]+box[3]/2)/608,box[2]/1088,box[3]/608])

            # bbox
            mask = torch.from_numpy(mask)
            mask = mask.float()
            mask = mask[None]
            mask = mask[None]
            mask = mask.contiguous()
            boxes = format_box(box)
            bbox = pass_box(box)
            box_index = torch.tensor([0], dtype=torch.int)

            # crop_height = int(bbox[3])
            # crop_width = int(bbox[2])
            # roi_align = RoIAlign(crop_height, crop_width, 0.25)

            crops = self.roi_align(mask, boxes, box_index)
            # print(boxes)
            # crops = torchvision.ops.roi_align(mask, boxes, (56, 56))[0]
            crops = crops.squeeze()
            mask_list.append(crops)
            bbox_list.append(bbox)
            # plt.imshow(crops)
            # plt.show()

        background_list = []

        for l in range(3):
            if l==0:
                levelbackground = cv2.resize(background, (68, 38))
                levelbackground = Image.fromarray(levelbackground.astype('uint8'))
                random.seed(seed)  # apply this seed to img tranfsorms
                levelbackground = self.transform(levelbackground)
                levelbackground = self.tr_image(levelbackground)
                background_list.append(levelbackground)
            elif l==1:
                levelbackground = cv2.resize(background, (136, 76))
                levelbackground = Image.fromarray(levelbackground.astype('uint8'))
                random.seed(seed)  # apply this seed to img tranfsorms
                levelbackground = self.transform(levelbackground)
                levelbackground = self.tr_image(levelbackground)
                background_list.append(levelbackground)
            elif l == 2:
                levelbackground = cv2.resize(background, (272, 152))
                levelbackground = Image.fromarray(levelbackground.astype('uint8'))
                random.seed(seed)  # apply this seed to img tranfsorms
                levelbackground = self.transform(levelbackground)
                levelbackground = self.tr_image(levelbackground)
                background_list.append(levelbackground)

        background = Image.fromarray(background.astype('uint8'))

        if self.transform is not None:
            random.seed(seed)  # apply this seed to img tranfsorms
            img = self.transform(img)
            img = self.tr_image(img)
            random.seed(seed)  # apply this seed to img tranfsorms
            background = self.transform(background)
            background = self.tr_image(background)
        return  {"img":img,"mask":background_list,"targets":torch.Tensor(targets),"targetslen":len(targets),"bbox":bbox_list,"mask_list":torch.cat(mask_list)}
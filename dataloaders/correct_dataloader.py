import cv2
from torch.utils.data import Dataset
from tools.mots_tools.io import *
import torch
from roi_align import RoIAlign

def format_box(bbox):
    return torch.Tensor([[bbox[0], bbox[1], bbox[0]+ bbox[2], bbox[1] + bbox[3]]])

def pass_box(bbox):
    return torch.Tensor([bbox[0], bbox[1], bbox[2], bbox[3]])

class MOTSCorrectDataset(Dataset):
    def __init__(self, inputRes=None,
                 seqs_list_file=r'E:\Challenge\MOTSChallenge\train\instances_txt',
                 transform=None,
                 sequence=2,
                 random_rev_thred=0.4):

        self.imgPath = os.path.join(r'E:\Challenge\MOTSChallenge\train\images',"{:04}".format(sequence))
        filename = os.path.join(seqs_list_file, "{:04}.txt".format(sequence))
        self.instance = load_txt(filename)
        self.transform = transform
        self.inputRes = inputRes
        self.random_rev_thred = random_rev_thred



    def __len__(self):
        return len(self.instance)

    def __getitem__(self, idx):
        frame=idx+1
        instance_per_frame = self.instance[frame]
        bbox_list = []
        track_list = []
        img = os.path.join(self.imgPath, "{:06}.jpg".format(frame))
        img = cv2.imread(img)
        img = cv2.resize(img, (2048, 1024))
        img = img[:,:,:].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        for obj in instance_per_frame:
            if obj.class_id!=2:
                continue
            track_list.append(obj.track_id%100)
            mask = rletools.decode(obj.mask)
            mask = cv2.resize(mask, (2048, 1024))
            newmask = np.asfortranarray(mask)
            newmask = newmask.astype(np.uint8)
            newmask = rletools.encode(newmask)
            # boxes = format_box(rletools.toBbox(obj.mask))
            bbox = pass_box(rletools.toBbox(newmask))

            bbox_list.append(bbox)

        if self.transform is not None:
            img = self.transform(img)
        return  {"img":img,"bbox":bbox_list,"track":track_list}
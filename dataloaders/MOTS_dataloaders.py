import cv2
from torch.utils.data import Dataset
from tools.mots_tools.io import *
import torch

def format_box(bbox):
    return torch.Tensor([[bbox[0], bbox[1], bbox[0]+ bbox[2], bbox[1] + bbox[3]]])

class MOTSDataset(Dataset):
    def __init__(self, inputRes=None,
                 seqs_list_file=r'E:\Challenge\MOTSChallenge\train\instances_txt',
                 transform=None,
                 sequence=2,
                 random_rev_thred=0.4):

        self.imgPath = os.path.join(r'E:\Challenge\MOTSChallenge\train\images', "{:04}.txt".format(sequence))
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
        instance_list = []
        for obj in instance_per_frame:
            mask = rletools.decode(obj.mask)
            mask = torch.from_numpy(mask)
            mask = mask.float()
            mask = mask[None]
            mask = mask[None]
            mask = mask.contiguous()

            img = os.path.join(self.imgPath, "{:06}.jpg".format(frame))
            img = cv2.imread(img)

            instance_list.append({
                "img":img,
                "mask": mask,
                "bbox": format_box(rletools.toBbox(obj.mask))
            })

        return  instance_list
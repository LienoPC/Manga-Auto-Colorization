import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
from PIL import Image
import torch


class ImageDataset(Dataset):

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        # Should be "empty"
        self.images = []
        self.transform = transform
        for img_path in os.listdir(img_dir):
            self.images.append(os.path.join(img_dir,img_path))
        ann_file = pd.DataFrame(self.images)
        ann_file.to_csv("Dataset/annotation.csv", header=False, index=False)
        self.images = pd.read_csv("Dataset/annotation.csv")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images.iloc[idx,0]
        image = self.load_img(str(img_path)) # Transform the image in tensor
        if self.transform:
            image = self.transform(image)
        return image

    @staticmethod
    def load_img(img_path):
        out_np = np.asarray(Image.open(img_path))
        if out_np.ndim == 2:
            out_np = np.tile(out_np[:, :, None], 3)
        return out_np




class LabNormalization():

    def __init__(self):
        self.l_channel_norm = 100
        self.ab_channel_norm = 110

    @staticmethod
    def get_l_channel(x):
        l_channel = x[:, :, 0].float()
        l_channel = l_channel / 255.0
        l_channel = l_channel.unsqueeze(0).unsqueeze(0)
        return l_channel

    @staticmethod
    def get_ab_channel(x):
        l_channel = x[:, :, 0].float()
        l_channel = l_channel / 255.0
        l_channel = l_channel.unsqueeze(0).unsqueeze(0)
        return l_channel

    @staticmethod
    def get_lab(x):
        lab_image = kornia.color.rgb_to_lab(x)
        return lab_image

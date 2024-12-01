import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
from PIL import Image
from skimage import color
import torch
import torch.nn.functional as F

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

    @staticmethod
    def res_image(image, new_size=(256, 256), resample=3):
        return np.asarray(Image.fromarray(image).resize((new_size[1], new_size[0]), resample=resample))

    @staticmethod
    def preprocess_img(img_rgb_orig, new_size=(256, 256), resample=3):

        img_rgb_rs = ImageDataset.res_image(img_rgb_orig, new_size=new_size, resample=resample)

        img_lab_orig = color.rgb2lab(img_rgb_orig)
        img_lab_rs = color.rgb2lab(img_rgb_rs)

        img_l_orig = img_lab_orig[:, :, 0]
        img_l_rs = img_lab_rs[:, :, 0]

        tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
        tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]

        return tens_orig_l, tens_rs_l

    @staticmethod
    def postprocess_tens(tens_orig_l, out_ab):
        # tens_orig_l 	1 x 1 x H_orig x W_orig
        # out_ab 		1 x 2 x H x W

        size_orig = tens_orig_l.shape[2:]
        size = out_ab.shape[2:]

        # call resize function if needed
        if size_orig[0] != size[0] or size_orig[1] != size[1]:
            out_ab_orig = F.interpolate(out_ab, size=size_orig, mode='bilinear')
        else:
            out_ab_orig = out_ab

        out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
        return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))


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

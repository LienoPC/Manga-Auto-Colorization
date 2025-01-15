import os
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch
from skimage import color

class ImageDataset(Dataset):

    def __init__(self, img_dir, resize=None, transform=None):
        self.img_dir = img_dir
        self.resize = resize
        self.transform = transform
        # Should be "empty"
        self.images = []

        for img_path in os.listdir(img_dir):
            full_path = os.path.join(img_dir, img_path)

            try:
                # Try to lead the image and verify if it's RGB
                with Image.open(full_path) as img:
                    if img.mode == "RGB":
                        self.images.append(full_path)

            except Exception as e:
                print(f"Error in image loading {full_path}: {e}")


        ann_file = pd.DataFrame(self.images)
        ann_file.to_csv("Dataset/annotation.csv", header=False, index=False)
        self.images = pd.read_csv("Dataset/annotation.csv")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images.iloc[idx,0]
        image = self.load_img(str(img_path))  # Transform the image in a tensor
        if self.resize:
            image = ImageProcess.res_image(image, self.resize)

        image = np.asarray(image).copy()  # Ensure the array is writable
        tens_img = ImageProcess.preprocess_img_alternative(image, new_size=self.resize)
        img_lab_orig = tens_img

        if (self.transform):
            img_lab_orig = self.transform(tens_img)

        img_l_rs = img_lab_orig[0, :, :]

        img_l_rs = img_l_rs.unsqueeze(0)
        return img_l_rs, img_lab_orig

    @staticmethod
    def load_img(img_path):
        out_np = np.asarray(Image.open(img_path))
        if out_np.ndim == 2:
            out_np = np.tile(out_np[:, :, None], 3)
        return out_np


class LABNormalization:
    def __init__(self, l_mean, l_std, ab_mean, ab_std):
        self.l_mean = l_mean
        self.l_std = l_std
        self.ab_mean = ab_mean
        self.ab_std = ab_std

    def normalize_l(self, in_l):
        return (in_l - self.l_mean) / self.l_std

    def unnormalize_l(self, in_l):
        return (in_l * self.l_std) + self.l_mean

    def normalize_ab(self, in_ab):
        return (in_ab - self.ab_mean) / self.ab_std

    def unnormalize_ab(self, in_ab):
        return (in_ab * self.ab_std) + self.ab_mean

    def normalize_lab_batch(self, in_lab):
        in_lab[:, 0, :, :] = self.normalize_l(in_lab[:, 0, :, :])
        in_lab[:, 1:, :, :] = self.normalize_ab(in_lab[:, 1:, :, :])
        return in_lab

    def unnormalize_lab_batch(self, in_lab):
        in_lab[:, 0, :, :] = self.unnormalize_l(in_lab[:, 0, :, :])
        in_lab[:, 1:, :, :] = self.unnormalize_ab(in_lab[:, 1:, :, :])
        return in_lab


class ImageProcess:



    @staticmethod
    def res_image(image, new_size=(256, 256), resample=3):
        # Convert to RGB from RGBA if necessary
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)

        image_rs = Image.fromarray(image.astype(np.uint8)).resize((new_size[1], new_size[0]), resample=resample)
        return np.asarray(image_rs)

    @staticmethod
    def preprocess_img(img_rgb_orig, new_size=(256, 256), resample=3):

        img_rgb_rs = ImageProcess.res_image(img_rgb_orig, new_size=new_size, resample=resample)

        img_lab_orig = color.rgb2lab(img_rgb_orig)
        img_lab_rs = color.rgb2lab(img_rgb_rs)

        img_l_orig = img_lab_orig[:, :, 0]
        img_l_rs = img_lab_rs[:, :, 0]

        tens_orig_l = torch.Tensor(img_l_orig)[None, :, :]
        tens_rs_l = torch.Tensor(img_l_rs)[None, :, :]
        return tens_orig_l, tens_rs_l

    @staticmethod
    def preprocess_img_alternative(img_rgb_orig, new_size=(256, 256), resample=3):

        img_rgb_rs = ImageProcess.res_image(img_rgb_orig, new_size=new_size, resample=resample)
        img_lab_rs = color.rgb2lab(img_rgb_rs)

        tens_res = torch.Tensor(img_lab_rs)[:, :, :].permute((2, 0, 1))

        return tens_res

    @staticmethod
    def postprocess_tens(tens_orig_l, out_ab):
        # tens_orig_l 	1 x 1 x H_orig x W_orig
        # out_ab 		1 x 2 x H x W

        if tens_orig_l.ndim == 3:  # Missing batch dimension
            tens_orig_l = tens_orig_l.unsqueeze(0)
        if out_ab.ndim == 3:  # Missing batch dimension
            out_ab = out_ab.unsqueeze(0)
        size_orig = tens_orig_l.shape[2:]
        size = out_ab.shape[2:]

        # call resize function if needed
        if size_orig[0] != size[0] or size_orig[1] != size[1]:
            out_ab_orig = F.interpolate(out_ab, size=size_orig, mode='bilinear')
        else:
            out_ab_orig = out_ab

        out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
        return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))

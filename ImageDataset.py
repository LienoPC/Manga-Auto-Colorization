import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
from PIL import Image
import torch
from skimage import color

from Network import ZhangColorizationNetwork


class ImageDataset(Dataset):

    def __init__(self, img_dir, resize=None):
        self.img_dir = img_dir
        self.resize = resize
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
                print(f"Errore nel caricamento dell'immagine {full_path}: {e}")


        ann_file = pd.DataFrame(self.images)
        ann_file.to_csv("Dataset/annotation.csv", header=False, index=False)
        self.images = pd.read_csv("Dataset/annotation.csv")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images.iloc[idx,0]
        image = self.load_img(str(img_path)) # Transform the image in tensor
        if self.resize:
            image = ZhangColorizationNetwork.res_image(image, self.resize)
        image = np.asarray(image).copy()  # Ensure the array is writable
        _, l_resized = ZhangColorizationNetwork.preprocess_img(image)
        img_lab_orig = torch.Tensor(color.rgb2lab(image))[:, :, :].permute((2,0,1))
        return l_resized, img_lab_orig

    @staticmethod
    def load_img(img_path):
        out_np = np.asarray(Image.open(img_path))
        if out_np.ndim == 2:
            out_np = np.tile(out_np[:, :, None], 3)
        return out_np





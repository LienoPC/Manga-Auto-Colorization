import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from skimage import color

import torch.nn as nn
import torch.nn.init as init

from AdversarialNetwork import Discriminator
from Evaluation import test_patch
from ImageDataset import ImageDataset, ImageProcess, LABNormalization
from Network import ZhangColorizationNetwork
from P2PDiscriminator import PatchGAN
from Utility import init_weights_he, zhang_train, plot_loss, adv_base_train, weights_init, adv_patch_train, \
    store_trained_model



def test_patch_main():
    # Avoid a memory leak caused by KMeans from scikit-learn when there are fewer chunks than available threads.
    os.environ['OMP_NUM_THREADS'] = '3'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n####### USED DEVICE ######\n{device}")
    l_mean = 73.07875061035156
    l_std = 28.479511260986328
    ab_mean = 3.4011828899383545
    ab_std = 14.096550941467285

    # Define normalization class basing on
    lab_normalization = LABNormalization(l_mean, l_std, ab_mean, ab_std)

    # Create dataset
    test_set = ImageDataset("../TestSet", resize=(256, 256))
    print(f"Training set size: {len(test_set)}")
    batch_size = 64
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=False)

    img_dim = 256
    gen_path = "ADV_PATCH_G_Epoch5/model.pth"
    disc_path = "ADV_PATCH_D_Epoch5/model.pth"
    file_gen, file_disc = test_patch(device, test_loader, lab_normalization, img_dim, gen_path, disc_path)


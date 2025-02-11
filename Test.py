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
from Evaluation import test_patch, test_adv, test_zhang
from ImageDataset import ImageDataset, ImageProcess, LABNormalization
from Network import ZhangColorizationNetwork
from P2PDiscriminator import PatchGAN
from Utility import init_weights_he, zhang_train, plot_loss, adv_base_train, weights_init, adv_patch_train, \
    store_trained_model

def store_test(list_err_file, model_name):
    save_dir = f"./SavedTests/{model_name}/"
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for (name, err_file) in list_err_file:
        err_file.seek(0)
        output_file = open(f"./SavedTests/{model_name}/{name}Test.txt", "w")
        for line in err_file:
            output_file.write(line)

        #os.remove(err_file)


def test_zhang_main():
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
    print(f"Test set size: {len(test_set)}")
    batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=False)

    img_dim = 256
    model_path = "SavedModels/Good/ZHANG_Epoch_25/checkpoint.pth"
    file_gen = test_zhang(device, test_loader, lab_normalization, img_dim, model_path)
    store_test([("Model", file_gen)], "Zhang")


def test_adv_main():
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
    print(f"Test set size: {len(test_set)}")
    batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=False)

    img_dim = 256
    gen_path = "SavedModels/Good/ADV_BASE_G_Epoch40/checkpoint.pth"
    disc_path = "SavedModels/Good/ADV_BASE_D_Epoch40/checkpoint.pth"
    file_gen, file_disc = test_adv(device, test_loader, lab_normalization, img_dim, gen_path, disc_path)
    store_test([("Generator", file_gen), ("Discriminator", file_disc)], "Adv")

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
    print(f"Test set size: {len(test_set)}")
    batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=False)

    img_dim = 256
    gen_path = "SavedModels/ADV_PATCH_G_Epoch40/checkpoint.pth"
    disc_path = "SavedModels/ADV_PATCH_D_Epoch40/checkpoint.pth"
    file_gen, file_disc = test_patch(device, test_loader, lab_normalization, img_dim, gen_path, disc_path)
    store_test([("Generator", file_gen), ("Discriminator", file_disc)], "Patch")


#test_patch_main()
file_train_g = open("SavedModels/Good/ADV_PATCH_G_Epoch40/GenTrain.txt", "r")
file_valid_g = open("SavedModels/Good/ADV_PATCH_G_Epoch40/GenValid.txt", "r")
file_train_d = open("SavedModels/Good/ADV_PATCH_D_Epoch40/DiscTrain.txt", "r")
file_valid_d = open("SavedModels/Good/ADV_PATCH_D_Epoch40/DiscValid.txt", "r")

plot_loss([file_train_g, file_train_d], "Train", True)
plot_loss([file_valid_g, file_valid_d], "Valid", True)
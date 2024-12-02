import torchvision
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision import models, transforms, datasets
from torchsummary import summary

from ImageDataset import ImageDataset
from Network import ZhangColorizationNetwork, zhang_train
import matplotlib.pyplot as plt
import kornia

def view_dataset_example(dataset):

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = dataset[i]
        plt.imshow(img.squeeze(), cmap=plt.cm.binary)
    plt.show()


if __name__ == '__main__':
    # Create dataset
    datasetTest = ImageDataset("TablesImages")

    train_loader = torch.utils.data.DataLoader(datasetTest)


    module = ZhangColorizationNetwork()

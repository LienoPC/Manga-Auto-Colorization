import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchsummary import summary

from ImageDataset import ImageDataset
from Network import ZhangColorizationNetwork, zhang_train


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
    # Avoid a memory leak caused by KMeans from scikit-learn when there are fewer chunks than available threads.
    os.environ['OMP_NUM_THREADS'] = '3'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Network creation
    module = ZhangColorizationNetwork()
    module.to(device)
    summary(module, (1, 224, 224))

    # Create dataset
    training_set = ImageDataset("../Dataset/500", resize=(224, 224))
    print(f"Training set size: {len(training_set)}")

    # Batch size
    batch_size = round(len(training_set) / 80)
    # Preparing indices for validation set
    indices = list(range(len(training_set)))

    # selected as get 20% of the train set
    split = int(np.floor(0.2 * len(training_set)))
    train_sample = SubsetRandomSampler(indices[:split])
    valid_sample = SubsetRandomSampler(indices[split:])

    # Define the data loader
    train_loader = torch.utils.data.DataLoader(training_set, sampler=train_sample, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(training_set, sampler=valid_sample, batch_size=batch_size)

    # Optimizer
    parameters_to_optimize = module.parameters()
    lr = 0.001
    num_epochs = 2
    optimizer = optim.Adam(parameters_to_optimize, lr=lr)

    zhang_train(module, train_loader, valid_loader, device=device, optimizer=optimizer)

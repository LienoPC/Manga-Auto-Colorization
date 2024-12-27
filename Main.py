import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchsummary import summary
from skimage import color
from torchvision import transforms

import torch.nn as nn
import torch.nn.init as init
from ImageDataset import ImageDataset, ImageNorm
from Network import ZhangColorizationNetwork, zhang_train


def init_weights_he(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He initialization
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

def view_dataset_example(dataset):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        _, image = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        image = color.lab2rgb(image)  # Convert from Lab to RGB
        plt.imshow(image)
    plt.show()


def view_lab_image(image):
    plt.figure()
    image = torch.Tensor(image)
    image = image.permute(1, 2, 0).numpy()
    image = color.lab2rgb(image)  # Convert from Lab to RGB
    plt.imshow(image)
    plt.show()


def get_mean_std(loader):
    channels_sum, channels_square_sum, num_batches = 0,0,0

    for _, lab_orig in loader:
        channels_sum += torch.mean(lab_orig, dim=[0,2,3])
        channels_square_sum += torch.mean(lab_orig**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_square_sum/num_batches - mean**2)**0.5

    return mean, std

if __name__ == '__main__':
    # Avoid a memory leak caused by KMeans from scikit-learn when there are fewer chunks than available threads.
    os.environ['OMP_NUM_THREADS'] = '3'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Network creation
    module = ZhangColorizationNetwork()
    module.to(device)
    module.apply(init_weights_he)


    #mean, std = torch.Tensor([73.0788, 2.1111, 4.6912]), torch.Tensor([28.4795, 11.8343, 15.9388])
    #transform = transforms.Normalize(mean, std, )

    # Create dataset
    training_set = ImageDataset("../ProvaDataset", resize=(256, 256))
    print(f"Training set size: {len(training_set)}")

    # Batch size
    #batch_size = round(len(training_set) / 6)
    batch_size = len(training_set)
    # Preparing indices for validation set
    indices = list(range(len(training_set)))

    #mean, std = get_mean_std(loader)


    # selected as get 20% of the train set
    split = int(np.floor(0.8 * len(training_set)))
    train_sample = SubsetRandomSampler(indices[:split])
    valid_sample = SubsetRandomSampler(indices[split:])

    # Define the data loader
    train_loader = torch.utils.data.DataLoader(training_set, sampler=train_sample, batch_size=batch_size, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(training_set, sampler=valid_sample, batch_size=batch_size, drop_last=False)

    print(f"Number of batches: {len(train_loader)}\n")

    # Iterate through batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}/{len(train_loader)}, Images Shape: {images.shape}, Labels Shape: {labels.shape}")

    # Optimizer
    parameters_to_optimize = module.parameters()
    lr = 0.001
    num_epochs = 200


    optimizer = optim.Adam(parameters_to_optimize, lr=lr)


    l_orig, img = training_set[random.randint(0, split)]

    rgb_img = ImageNorm.postprocess_tens(l_orig, img[1:,:,:])
    plt.imshow(rgb_img)
    plt.show()

    print(f"Batch size: {train_loader.batch_size}")

    train_losses, valid_losses = zhang_train(module, train_loader, valid_loader, device=device, optimizer=optimizer, epochs=num_epochs)

    l_orig = l_orig.unsqueeze(0).to(device)
    conv8, ab_channel = module(l_orig)
    rgb_img_out = ImageNorm.postprocess_tens(l_orig, ab_channel)
    print("\n\n\nComputed AB channel:\n")
    print(ab_channel)
    plt.figure()
    plt.imshow(rgb_img_out)
    plt.show()

    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
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

from AdversarialNetwork import Discriminator, adv_train
from ImageDataset import ImageDataset, ImageProcess, LABNormalization
from Network import ZhangColorizationNetwork, zhang_train


# He weights initialization
def init_weights_he(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)


# Weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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


def get_lab_mean_std(loader):
    l_sum, l_square_sum = 0, 0
    ab_sum, ab_square_sum = 0, 0
    l_pixels, ab_pixels = 0, 0

    for _, lab_orig in loader:
        # Assuming lab_orig has shape [batch_size, 3, height, width]
        l_channel = lab_orig[:, 0, :, :]  # Extract L channel
        ab_channels = lab_orig[:, 1:, :, :]  # Extract AB channels

        # Compute for L channel
        l_sum += l_channel.sum()
        l_square_sum += (l_channel**2).sum()
        l_pixels += l_channel.numel()

        # Compute for AB channels (flatten A and B into one dimension)
        ab_sum += ab_channels.sum()
        ab_square_sum += (ab_channels**2).sum()
        ab_pixels += ab_channels.numel()

    # Mean and std for L channel
    l_mean = l_sum / l_pixels
    l_std = ((l_square_sum / l_pixels) - l_mean**2).sqrt()

    # Combined mean and std for AB channels
    ab_mean = ab_sum / ab_pixels
    ab_std = ((ab_square_sum / ab_pixels) - ab_mean**2).sqrt()

    return (l_mean, l_std), (ab_mean, ab_std)


# Main that uses only the Zhang model without the adversarial loss
'''
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
    num_epochs = 10


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
'''


class LabNormalization:
    pass


if __name__ == '__main__':
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

    # Networks creation
    module = ZhangColorizationNetwork(lab_normalization)
    module.to(device)
    discriminator = Discriminator()
    discriminator.to(device)

    # If needed we can apply an initialization to the weights of the networks
    # module.apply(init_weights_he)
    # discriminator.apply(weights_init)

    # l_mean: 73.07875061035156; l_std: 28.479511260986328; ab_mean: 3.4011828899383545; ab_std: 14.096550941467285

    # Create dataset
    dataset = ImageDataset("../ProvaDataset", resize=(256, 256))
    print(f"Training set size: {len(dataset)}")

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False)

    # Batch size
    batch_size = round(len(dataset) / 3)
    # batch_size = len(training_set)
    # Preparing indices for validation set
    indices = list(range(len(dataset)))

    # mean, std = get_mean_std(loader)

    # Selected as get 20% of the train set
    split = int(np.floor(0.8 * len(dataset)))
    train_sample = SubsetRandomSampler(indices[:split])
    valid_sample = SubsetRandomSampler(indices[split:])

    # Define the data loader
    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sample, batch_size=batch_size, drop_last=False, pin_memory=False, persistent_workers=False)
    valid_loader = torch.utils.data.DataLoader(dataset, sampler=valid_sample, batch_size=batch_size, drop_last=False, pin_memory=False, persistent_workers=False)

    print(f"Number of batches: {len(train_loader)}\n")
    print(f"Batch size: {train_loader.batch_size}")
    # Optimizer
    gen_parameters_to_optimize = module.parameters()
    disc_parameters_to_optimize = discriminator.parameters()
    lr_gen = 0.0002
    lr_disc = 0.00002
    num_epochs = 8

    gen_optimizer = optim.Adam(gen_parameters_to_optimize, lr=lr_gen, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(disc_parameters_to_optimize, lr=lr_disc, betas=(0.5, 0.999))

    # Extract a random image from the validation set to evaluate the model after the training
    l_orig, img = dataset[random.randint(split, len(dataset)-1)]

    # Show the colored, ground-truth, image
    rgb_img = ImageProcess.postprocess_tens(l_orig, img[1:, :, :])
    plt.imshow(rgb_img)
    plt.show()

    # Train the model(s)
    file_gen, file_disc = adv_train(module, discriminator, train_loader, valid_loader, lab_normalization=lab_normalization,device=device, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer, epochs=num_epochs)

    # Take the saved values of losses from the disk and plot them
    file_gen.seek(0)
    file_disc.seek(0)

    fig, ax = plt.subplots()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # Initialize data arrays for both generators and discriminators
    x_gen_values = []
    y_gen_values = []
    x_disc_values = []
    y_disc_values = []
    # Loop through both files
    for l_gen, l_disc in zip(file_gen, file_disc):
        x_gen, y_gen = map(float, l_gen.strip().split(","))
        x_disc, y_disc = map(float, l_disc.strip().split(","))

        #print(f"x_gen, x_disc: {x_gen, x_disc}\n")
        #print(f"y_gen, y_disc: {y_gen, y_disc}\n")

        # Append new points to the arrays
        x_gen_values.append(x_gen)
        y_gen_values.append(y_gen)
        x_disc_values.append(x_disc)
        y_disc_values.append(y_disc)

        # Create two lines for generator and discriminator
    line_gen, = ax.plot(x_gen_values, y_gen_values, 'o-', label="Generator")
    line_disc, = ax.plot(x_disc_values, y_disc_values, 'x-', label="Discriminator")
    plt.show()
    file_gen.close()
    file_disc.close()

    '''
    l_orig = l_orig.unsqueeze(0).to(device)
    _, ab_channel = module(l_orig)
    rgb_img_out = ImageProcess.postprocess_tens(l_orig, ab_channel)
    plt.figure()
    plt.imshow(rgb_img_out)
    plt.show()
    '''


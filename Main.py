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
from ImageDataset import ImageDataset, ImageProcess, LABNormalization
from Network import ZhangColorizationNetwork
from P2PDiscriminator import PatchGAN
from Utility import init_weights_he, zhang_train, plot_loss, adv_base_train, weights_init, adv_patch_train, \
    store_trained_model


# Main that uses only the Zhang model without the adversarial loss

def zhang_model_main():
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
    # Network creation
    module = ZhangColorizationNetwork(lab_normalization)
    module.to(device)
    module.apply(init_weights_he)


    #mean, std = torch.Tensor([73.0788, 2.1111, 4.6912]), torch.Tensor([28.4795, 11.8343, 15.9388])
    #transform = transforms.Normalize(mean, std, )

    # Create dataset
    training_set = ImageDataset("../Dataset", resize=(256, 256))
    print(f"Training set size: {len(training_set)}")

    # Batch size
    batch_size = 16
    #batch_size = len(training_set)
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

    # Optimizer
    parameters_to_optimize = module.parameters()
    lr = 0.001
    num_epochs = 8


    optimizer = optim.Adam(parameters_to_optimize, lr=lr)


    l_orig, img = training_set[random.randint(0, split)]

    rgb_img = ImageProcess.postprocess_tens(l_orig, img[1:,:,:])
    plt.imshow(rgb_img)
    plt.show()

    print(f"Batch size: {train_loader.batch_size}")

    train_loss_file, valid_loss_file = zhang_train(module, train_loader, valid_loader, device=device, optimizer=optimizer, lab_normalization=lab_normalization, epochs=num_epochs)

    train_loss_file.seek(0)
    valid_loss_file.seek(0)
    plot_loss([train_loss_file], "Train Loss", False)
    l_orig = l_orig.unsqueeze(0).to(device)
    conv8, ab_channel = module(l_orig)
    rgb_img_out = ImageProcess.postprocess_tens(l_orig, ab_channel)
    plt.figure()
    plt.imshow(rgb_img_out)
    plt.show()
    store_trained_model(module, [("GenTrain",train_loss_file), ("DiscTrain",valid_loss_file)], "ZHANG")







def adv_base_model_main():
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
    dataset = ImageDataset("../Dataset", resize=(256, 256))
    print(f"Training set size: {len(dataset)}")

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False)

    # Batch size
    batch_size = 4
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
    file_train_g, file_train_d, file_valid_g, file_valid_d = adv_base_train(module, discriminator, train_loader, valid_loader, img_dim=256, lab_normalization=lab_normalization,device=device, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer, epochs=num_epochs)

    # Take the saved values of losses from the disk and plot them
    file_train_g.seek(0)
    file_train_d.seek(0)

    plot_loss([file_train_g, file_train_d], "Train", True)
    plot_loss([file_valid_g, file_valid_d], "Train", True)

    l_orig = l_orig.unsqueeze(0).to(device)
    _, ab_channel = module(l_orig)
    rgb_img_out = ImageProcess.postprocess_tens(l_orig, ab_channel)
    plt.figure()
    plt.imshow(rgb_img_out)
    plt.show()

    store_trained_model(module, [("GenTrain",file_train_g), ("DiscTrain",file_train_d), ("GenValid",file_valid_g), ("DiscValid",file_valid_d)], "ADV_B")
  




# ADV PATCH TRAINING
def adv_patch_model_main():
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
    discriminator = PatchGAN(3)
    discriminator.to(device)

    # If needed we can apply an initialization to the weights of the networks
    module.apply(init_weights_he)
    discriminator.apply(weights_init)

    # l_mean: 73.07875061035156; l_std: 28.479511260986328; ab_mean: 3.4011828899383545; ab_std: 14.096550941467285

    # Create dataset
    dataset = ImageDataset("../Dataset", resize=(256, 256))
    print(f"Training set size: {len(dataset)}")

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True)

    # Batch size
    batch_size = 8
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
    file_train_g, file_train_d, file_valid_g, file_valid_d = adv_patch_train(module, discriminator, train_loader, valid_loader, lab_normalization=lab_normalization,device=device, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer, epochs=num_epochs, img_dim=256)

    # Take the saved values of losses from the disk and plot them
    file_train_g.seek(0)
    file_train_d.seek(0)

    plot_loss([file_train_g, file_train_d], "Train", True)
    plot_loss([file_valid_g, file_valid_d], "Valid", True)


    l_orig = l_orig.unsqueeze(0).to(device)
    _, ab_channel = module(l_orig)
    rgb_img_out = ImageProcess.postprocess_tens(l_orig, ab_channel)
    plt.figure()
    plt.imshow(rgb_img_out)
    plt.show()
    store_trained_model(module, [("GenTrain",file_train_g), ("DiscTrain",file_train_d), ("GenValid",file_valid_g), ("DiscValid",file_valid_d)], "ADV_PATCH")



adv_base_model_main()
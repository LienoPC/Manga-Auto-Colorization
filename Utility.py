import os
import random
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from skimage import color

import torch.nn as nn
import torch.nn.init as init

from AdversarialNetwork import adv_train_step, adv_valid_step
from Network import ZhangColorizationNetwork, quantized_bins, zhang_train_step
from P2PDiscriminator import PatchGAN, adv_patch_train_step, adv_patch_valid_step


def adv_patch_train(generator, discriminator, trainloader, validloader, device, gen_optimizer, disc_optimizer, lab_normalization, img_dim, epoch, epochs=2):
    """
      Args:
        generator: The model of the generator to train.
        discriminator: The model of the discriminator.
        trainloader: The DataLoader for the training set.
        validloader: The DataLoader for the validation set.
        device: The device used for training.
        gen_optimizer: Optimizer used to train the generator.
        disc_optimizer: Optimizer used to train the discriminator.
        lab_normalization: Object that manages lab normalization.
        epochs: The number of training epochs.
        learning_rate: The learning rate for the optimizer.
      """


    # Compute the quantized bins and move them to the correct device
    quantized_colorspace = quantized_bins().to(device)
    print("Started training...")

    temp_file_train_g = tempfile.NamedTemporaryFile(mode="w+")
    temp_file_train_d = tempfile.NamedTemporaryFile(mode="w+")

    temp_file_valid_g = tempfile.NamedTemporaryFile(mode="w+")
    temp_file_valid_d = tempfile.NamedTemporaryFile(mode="w+")


    for epoch in range(epoch, epochs):

        temp_file_train_g, temp_file_train_d, gen_train_loss, disc_train_loss = adv_patch_train_step(generator, discriminator, trainloader, device, gen_optimizer, disc_optimizer, lab_normalization, temp_file_train_g, temp_file_train_d, quantized_colorspace, img_dim, epoch)
        temp_file_train_g.flush()
        temp_file_train_d.flush()
        temp_file_valid_g, temp_file_valid_d, gen_valid_loss, disc_valid_loss = adv_patch_valid_step(generator, discriminator, validloader, device, gen_optimizer, disc_optimizer, lab_normalization, temp_file_valid_g, temp_file_valid_d, quantized_colorspace, img_dim, epoch)
        temp_file_valid_g.flush()
        temp_file_valid_d.flush()

        print(f"Epoch [{epoch}/{epochs}], Gen Train Loss: {gen_train_loss:.4f}, Disc Train Loss: {disc_train_loss:.4f};   Gen Valid Loss: {gen_valid_loss:.4f}, Disc Valid Loss: {disc_valid_loss:.4f}")
        del gen_train_loss, disc_train_loss, gen_valid_loss, disc_valid_loss
        store_trained_model(generator,[("GenTrain", temp_file_train_g), ("GenValid", temp_file_valid_g)], f"ADV_PATCH_G_Epoch{epoch}",epoch=epoch, optimizer=gen_optimizer)
        store_trained_model(discriminator,
                            [("DiscTrain", temp_file_train_d),
                             ("DiscValid", temp_file_valid_d)], f"ADV_PATCH_D_Epoch{epoch}",epoch=epoch, optimizer=disc_optimizer)

    return temp_file_train_g, temp_file_train_d, temp_file_valid_g, temp_file_valid_d


def adv_base_train(generator, discriminator, trainloader, validloader, device, gen_optimizer, disc_optimizer, lab_normalization, img_dim, epoch, epochs=2):
    """
      Args:
        generator: The model of the generator to train.
        discriminator: The model of the discriminator.
        trainloader: The DataLoader for the training set.
        validloader: The DataLoader for the validation set.
        device: The device used for training.
        gen_optimizer: Optimizer used to train the generator.
        disc_optimizer: Optimizer used to train the discriminator.
        lab_normalization: Object that manages lab normalization.
        epochs: The number of training epochs.
        learning_rate: The learning rate for the optimizer.
      """


    # Compute the quantized bins and move them to the correct device
    quantized_colorspace = quantized_bins().to(device)
    print("Started training...")

    temp_file_train_g = tempfile.NamedTemporaryFile(mode="w+")
    temp_file_train_d = tempfile.NamedTemporaryFile(mode="w+")
    temp_file_valid_g = tempfile.NamedTemporaryFile(mode="w+")
    temp_file_valid_d = tempfile.NamedTemporaryFile(mode="w+")


    for epoch in range(epoch, epochs):

        temp_file_train_g, temp_file_train_d, gen_train_loss, disc_train_loss = adv_train_step(generator, discriminator, trainloader, device, gen_optimizer, disc_optimizer, lab_normalization, temp_file_train_g, temp_file_train_d, quantized_colorspace, epoch)
        temp_file_train_g.flush()
        temp_file_train_d.flush()
        temp_file_valid_g, temp_file_valid_d, gen_valid_loss, disc_valid_loss = adv_valid_step(generator, discriminator, validloader, device, gen_optimizer, disc_optimizer, lab_normalization, temp_file_valid_g, temp_file_valid_d, quantized_colorspace, epoch)
        temp_file_valid_g.flush()
        temp_file_valid_d.flush()
        print(f"Epoch [{epoch}/{epochs}], Gen Train Loss: {gen_train_loss:.4f}, Disc Train Loss: {disc_train_loss:.4f};   Gen Valid Loss: {gen_valid_loss:.4f}, Disc Valid Loss: {disc_valid_loss:.4f}")
        del gen_train_loss, disc_train_loss, gen_valid_loss, disc_valid_loss
        store_trained_model(generator,
                            [("GenTrain", temp_file_train_g), ("DiscTrain", temp_file_train_d), ("GenValid", temp_file_valid_g),
                             ("DiscValid", temp_file_valid_d)], f"ADV_BASE_G_Epoch{epoch}", epoch=epoch, optimizer=gen_optimizer)
        store_trained_model(discriminator,
                        [("DiscTrain", temp_file_train_d),
                         ("DiscValid", temp_file_valid_d)], f"ADV_BASE_D_Epoch{epoch}",epoch=epoch, optimizer=disc_optimizer)

    return temp_file_train_g, temp_file_train_d, temp_file_valid_g, temp_file_valid_d


def zhang_train(model, trainloader, validloader, device, optimizer, lab_normalization, epoch, epochs=2):
    """
          Trains a PyTorch model and returns the training and validation losses.

          Args:
            model: The PyTorch model to train.
            trainloader: The DataLoader for the training set.
            validloader: The DataLoader for the validation set.
            epochs: The number of training epochs.
            learning_rate: The learning rate for the optimizer.

          Returns:
            train_losses: A list of training losses for each epoch.
            valid_losses: A list of validation losses for each epoch.
          """

    # Compute the quantized bins and move them to the correct device
    quantized_colorspace = quantized_bins().to(device)
    print("Started training...")
    temp_file_train = tempfile.NamedTemporaryFile(mode="w+")
    temp_file_valid = tempfile.NamedTemporaryFile(mode="w+")
    for epoch in range(epoch,epochs):
        temp_file_train, train_loss = zhang_train_step(model, trainloader, device,
                                                              optimizer, lab_normalization,
                                                              temp_file_train,
                                                              quantized_colorspace, epoch, True)

        temp_file_valid, valid_loss = zhang_train_step(model, validloader, device,
                                                              optimizer, lab_normalization,
                                                              temp_file_valid,
                                                              quantized_colorspace, epoch, True)

        print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        # del gen_train_loss, gen_valid_loss, gen_valid_loss, disc_valid_loss
        store_trained_model(model, [("Train",temp_file_train), ("Valid",temp_file_valid)], f"ZHANG_Epoch_{epoch}", epoch=epoch, optimizer=optimizer)

    return temp_file_train, temp_file_valid


# Plot loss over epochs
def plot_loss(files, label, adversarial=True):
    if adversarial:
        file_gen = files[0]
        file_disc = files[1]
        # Take the saved values of losses from the disk and plot them
        file_gen.seek(0)
        file_disc.seek(0)

        fig, ax = plt.subplots()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # Initialize data arrays for both generators and discriminators
        x_gen_values = []
        y_gen_values = []
        x_disc_values = []
        y_disc_values = []
        # Loop through both files
        for l_gen, l_disc in zip(file_gen, file_disc):
            x_gen, y_gen = map(float, l_gen.strip().split(","))
            x_disc, y_disc = map(float, l_disc.strip().split(","))

            # print(f"x_gen, x_disc: {x_gen, x_disc}\n")
            # print(f"y_gen, y_disc: {y_gen, y_disc}\n")

            # Append new points to the arrays
            x_gen_values.append(x_gen)
            y_gen_values.append(y_gen)
            x_disc_values.append(x_disc)
            y_disc_values.append(y_disc)

            # Create two lines for generator and discriminator
        line_gen, = ax.plot(x_gen_values, y_gen_values, 'o-', label=f"Generator {label}")
        line_disc, = ax.plot(x_disc_values, y_disc_values, 'x-', label=f"Discriminator {label}")
        plt.legend()
        plt.show()
    else:
        file = files[0]
        # Take the saved values of losses from the disk and plot them
        file.seek(0)

        fig, ax = plt.subplots()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # Initialize data arrays for both generators and discriminators
        x_values = []
        y_values = []
        # Loop through both files
        for line in file:
            x, y = map(float, line.strip().split(","))
            # print(f"x_gen, x_disc: {x_gen, x_disc}\n")
            # print(f"y_gen, y_disc: {y_gen, y_disc}\n")

            # Append new points to the arrays
            x_values.append(x)
            y_values.append(y)


            # Create two lines for generator and discriminator
        line_gen, = ax.plot(x_values, y_values, 'o-', label=f"{label}")
        plt.legend()
        plt.show()


# Store trained model, save training error
def store_trained_model(model, list_err_file, model_name, epoch, optimizer):
    save_dir = f"./SavedModels/{model_name}/"
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'{save_dir}/checkpoint.pth')
    for (name, err_file) in list_err_file:
        err_file.seek(0)
        output_file = open(f"./SavedModels/{model_name}/{name}.txt", "w")
        for line in err_file:
            output_file.write(line)

        #os.remove(err_file)


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


# Plot some examples
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

# Plot lab image
def view_lab_image(image):
    plt.figure()
    image = torch.Tensor(image)
    image = image.permute(1, 2, 0).numpy()
    image = color.lab2rgb(image)  # Convert from Lab to RGB
    plt.imshow(image)
    plt.show()

# Compute mean of lab batch images
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
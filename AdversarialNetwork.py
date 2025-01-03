import torchvision
import torch
import torch.nn as nn
import numpy as np
from skimage import color
from PIL import Image
import torch.nn.functional as F
from skimage.color import lab2rgb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from ImageDataset import ImageNorm
from Network import quantized_bins, resize_to_64x64, inverse_h_mapping, multinomial_cross_entropy_loss_L


def plot_batch_images(batch, title="Batch Images", nrow=8, figsize=(12, 12), normalize=True):
    batch = ImageNorm.unnormalize_lab_batch(batch)
    batch_rgb = []
    for lab_img in batch:
        rgb_img = color.lab2rgb(lab_img.permute(1, 2, 0))
        batch_rgb.append(rgb_img)
        # Convert the batch to a grid

    batch_rgb = np.stack(batch_rgb)  # Stack back to batch shape (B, H, W, C)
    batch_rgb_tensor = torch.tensor(batch_rgb).permute(0, 3, 1, 2)  # Convert to (B, C, H, W) for make_grid
    grid = make_grid(batch_rgb_tensor, nrow=nrow, normalize=False, value_range=(0, 1))
    np_grid = grid.permute(1, 2, 0).numpy()  # Rearrange for Matplotlib
    # Plot the grid
    plt.figure(figsize=figsize)
    plt.imshow(np_grid)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2), nn.BatchNorm2d(64), nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.BatchNorm2d(128), nn.LeakyReLU(0.1))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2), nn.BatchNorm2d(256), nn.LeakyReLU(0.1))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, stride=2), nn.BatchNorm2d(512), nn.LeakyReLU(0.1))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=14, stride=1, padding=0), nn.LeakyReLU(0.1))
        self.conv6 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        # Starting from the image
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x.view(x.size(0), -1)


def adv_train(generator, discriminator, trainloader, validloader, device, gen_optimizer, disc_optimizer, epochs=2):
    """
      Trains a PyTorch model and returns the training and validation losses.

      Args:
        generator: The PyTorch model to train.
        trainloader: The DataLoader for the training set.
        validloader: The DataLoader for the validation set.
        epochs: The number of training epochs.
        learning_rate: The learning rate for the optimizer.

      Returns:
        train_losses: A list of training losses for each epoch.
        valid_losses: A list of validation losses for each epoch.
      """

    DISCRIMINATOR_LOSS_THRESHOLD = 0.7
    # Compute the quantized bins and move them to the correct device
    quantized_colorspace = quantized_bins().to(device)
    print("Started training...")
    adv_loss_criterion = nn.BCELoss()
    pixel_loss_criterion = nn.MSELoss()

    gen_train_losses = []
    disc_train_losses = []
    # Setup discriminator application and loss computation


    for epoch in range(epochs):

        # Set the model to training mode
        generator.train()
        discriminator.train()
        r_gen_loss = 0.0
        r_disc_loss = 0.0
        for batch_idx, (l_resized, img_lab_orig) in enumerate(trainloader):
            target_truth = torch.ones(l_resized.shape[0], 1) * 0.9  # Smoothed labels for real data
            target_false = torch.zeros(l_resized.shape[0], 1) + 0.1  # Smoothed labels for fake data

            target_truth = target_truth.to(device)
            target_false = target_false.to(device)

            # Get the LAB original image
            l_resized = l_resized.to(device)
            img_lab_orig = img_lab_orig.to(device)

            # Extract the ground truth ab and convert it to tensor
            ab_groundtruth = img_lab_orig[:, 1:3, :, :]
            # Normalize the AB groundtruth
            ab_groundtruth = ImageNorm.normalize_ab(ab_groundtruth)

            # Normalize the LAB original image
            img_lab_orig = ImageNorm.normalize_lab_batch(img_lab_orig)
            # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
            ab_groundtruth = resize_to_64x64(ab_groundtruth)
            z_ground, _ = inverse_h_mapping(ab_groundtruth, quantized_colorspace)

            # 1. Train the Discriminator
            disc_optimizer.zero_grad()
            for param in generator.parameters():
                param.requires_grad = False
            for param in discriminator.parameters():
                param.requires_grad = True

            with torch.no_grad():  # Freeze the generator for discriminator training
                raw_conv8_output, ab_output = generator(l_resized)
                gen_lab_out = torch.cat((l_resized, ab_output), dim=1)
                gen_lab_out = ImageNorm.normalize_lab_batch(gen_lab_out)

            disc_ground = discriminator(img_lab_orig)
            disc_ground_loss = adv_loss_criterion(disc_ground, target_truth)

            disc_gen = discriminator(gen_lab_out)
            disc_gen_loss = adv_loss_criterion(disc_gen, target_false)

            disc_loss = disc_gen_loss + disc_ground_loss
            print(f"Discriminator Real Prediction: {disc_ground.mean().item()}, Fake Prediction: {disc_gen.mean().item()}")
            # Check if the loss is above the threshold
            if disc_loss.item() > DISCRIMINATOR_LOSS_THRESHOLD:
                disc_ground_loss.backward()
                disc_gen_loss.backward()  # Perform backpropagation only if the condition is met
                disc_optimizer.step()  # Update the discriminator's parameters
            else:
                print(f"Skipping discriminator update. Loss: {disc_loss.item():.4f}")


            # 2. Train the Generator
            gen_optimizer.zero_grad()
            for param in generator.parameters():
                param.requires_grad = True
            for param in discriminator.parameters():
                param.requires_grad = False

            raw_conv8_output, ab_output = generator(l_resized)  # Perform a new forward pass
            gen_lab_out = torch.cat((l_resized, ab_output), dim=1)
            gen_lab_out = ImageNorm.normalize_lab_batch(gen_lab_out)
            disc_gen = discriminator(gen_lab_out)  # Discriminator output for generated images
            '''
            print("Generator output (ab_output):", ab_output.mean().item(), ab_output.std().item())
            print("Discriminator output (ground):", disc_ground.mean().item(), disc_ground.std().item())
            print("Discriminator output (generated):", disc_gen.mean().item(), disc_gen.std().item())
            '''
            # Generator loss combines adversarial loss, the Z-space loss and the pixel loss
            adv_loss = adv_loss_criterion(disc_gen, target_truth)  # Fool the discriminator
            z_loss = multinomial_cross_entropy_loss_L(raw_conv8_output, z_ground_truth=z_ground)
            pixel_loss = pixel_loss_criterion(gen_lab_out, img_lab_orig)

            gen_loss = adv_loss + z_loss + pixel_loss

            gen_loss.backward()
            gen_optimizer.step()

            r_gen_loss += gen_loss
            r_disc_loss += disc_loss

            plot_batch_images(gen_lab_out.detach().to("cpu"))

            # Cleanup
            del l_resized, img_lab_orig, ab_groundtruth, z_ground, raw_conv8_output, ab_output
            torch.cuda.empty_cache()  # Clear CUDA cache explicitly
            '''
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Generator param {name} grad mean: {param.grad.mean().item()}")

            for name, param in discriminator.named_parameters():
                if param.grad is not None:
                    print(f"Discriminator param {name} grad mean: {param.grad.mean().item()}")

            '''

        gen_train_loss = r_gen_loss
        disc_train_loss = r_disc_loss
        # Validation
        generator.eval()  # Set the model to evaluation mode
        discriminator.eval()
        r_gen_loss = 0.0
        r_disc_loss = 0.0
        with torch.no_grad():
            for l_resized_val, img_lab_orig_val in validloader:
                target_truth = torch.ones(l_resized_val.shape[0], 1) * 0.9  # Smoothed labels for real data
                target_false = torch.zeros(l_resized_val.shape[0], 1) + 0.1  # Smoothed labels for fake data
                target_truth = target_truth.to(device)
                target_false = target_false.to(device)

                l_resized_val = l_resized_val.to(device)
                img_lab_orig_val = img_lab_orig_val.to(device)

                # Extract the ground truth ab and convert it to tensor
                ab_groundtruth = img_lab_orig_val[:, 1:3, :, :]
                # Normalize the AB groundtruth
                ab_groundtruth = ImageNorm.normalize_ab(ab_groundtruth)

                # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
                ab_groundtruth = resize_to_64x64(ab_groundtruth)
                z_ground, _ = inverse_h_mapping(ab_groundtruth, quantized_colorspace)

                # Calculate the final generated image
                raw_conv8_output, ab_output = generator(l_resized_val)
                gen_lab_out = torch.cat((l_resized_val, ab_output), dim=1)



                # Apply discriminator over ground_truth
                disc_ground = discriminator(img_lab_orig_val)
                disc_ground = disc_ground.to(device)
                disc_ground_loss = adv_loss_criterion(disc_ground, target_truth)

                disc_gen = discriminator(gen_lab_out)
                disc_gen_loss = adv_loss_criterion(disc_gen, target_false)

                disc_loss = disc_gen_loss + disc_ground_loss

                r_disc_loss += disc_loss.item()

                # Compute the custom loss over the Z space
                loss = multinomial_cross_entropy_loss_L(raw_network_output=raw_conv8_output, z_ground_truth=z_ground)
                r_gen_loss += loss.item()
                # print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(trainloader)}], Loss: {loss.item()}")

                # Cleanup
                del l_resized_val, img_lab_orig_val, ab_groundtruth, z_ground, raw_conv8_output, ab_output
                torch.cuda.empty_cache()  # Clear CUDA cache explicitly


        gen_valid_loss = r_gen_loss
        disc_valid_loss = r_disc_loss
        print(f"Epoch [{epoch + 1}/{epochs}], Gen Train Loss: {gen_train_loss:.4f}, Disc Train Loss: {disc_train_loss:.4f};   Gen Valid Loss: {gen_valid_loss:.4f}, Disc Valid Loss: {disc_valid_loss:.4f}")
        #del gen_train_loss, gen_valid_loss, gen_valid_loss, disc_valid_loss



import torch
import torch.nn as nn
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from torch import Tensor
from torch.autograd import Variable
from torchvision.utils import make_grid
import tempfile

from AdversarialNetwork import plot_batch_images
from Network import quantized_bins, resize_to_64x64, inverse_h_mapping, multinomial_cross_entropy_loss_L

# Discriminator based on a patch of the input and the classification of sub-sections of the image
# The final version proposed by the Pix2Pix architecture considers a 70x70 patch
class PatchGAN(nn.Module):

    def __init__(self, in_channel):
        super(PatchGAN,self).__init__()
        # Modify the patchGAN in order to work on L channel as input
        self.cd64 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
                                  nn.BatchNorm2d(64), nn.Dropout2d(0.5), nn.LeakyReLU(0.2, inplace=True))
        self.cd128 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                  nn.InstanceNorm2d(128), nn.Dropout2d(0.5), nn.LeakyReLU(0.2, inplace=True))
        self.cd256 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                                  nn.InstanceNorm2d(256), nn.Dropout2d(0.5), nn.LeakyReLU(0.2, inplace=True))
        self.cd512 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                                  nn.InstanceNorm2d(512), nn.Dropout2d(0.5), nn.LeakyReLU(0.2, inplace=True))
        self.out = nn.Sequential(nn.ZeroPad2d((1,0,1,0)), nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False), nn.Sigmoid())

    def forward(self, bw_image, col_image):

        img_input = torch.cat((bw_image,col_image), 1)
        # Starting from the image
        img_input = self.cd64(img_input)
        img_input = self.cd128(img_input)
        img_input = self.cd256(img_input)
        img_input = self.cd512(img_input)
        out = self.out(img_input)
        return out


def plot_batch_bw(batch, lab_normalization, title="BW LAB Training Images", nrow=8, figsize=(12, 12), normalize=True):
    batch = lab_normalization.unnormalize_lab_batch(batch)
    batch_rgb = []
    for lab_img in batch:
        rgb_img = color.lab2rgb(lab_img.permute(1, 2, 0))
        batch_rgb.append(rgb_img)

    batch_rgb = np.stack(batch_rgb)
    batch_rgb_tensor = torch.tensor(batch_rgb).permute(0, 3, 1, 2)
    grid = make_grid(batch_rgb_tensor, nrow=nrow, normalize=False, value_range=(0, 1))
    np_grid = grid.permute(1, 2, 0).numpy()  # Rearrange for Matplotlib
    # Plot the grid
    plt.figure(figsize=figsize)
    plt.imshow(np_grid)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()



def adv_patch_train(generator, discriminator, trainloader, validloader, device, gen_optimizer, disc_optimizer, lab_normalization, img_dim, epochs=2):
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

    DISCRIMINATOR_LOSS_THRESHOLD = 0.4
    PIXEL_FACTOR = 1.0
    Z_LOSS_FACTOR = 3.0
    # Compute the quantized bins and move them to the correct device
    quantized_colorspace = quantized_bins().to(device)
    print("Started training...")
    adv_loss_criterion = nn.MSELoss()
    pixel_loss_criterion = nn.L1Loss()

    temp_file_generator = tempfile.TemporaryFile(mode="w+")
    temp_file_discriminator = tempfile.TemporaryFile(mode="w+")

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, img_dim // 2 ** 4, img_dim // 2 ** 4)
    for epoch in range(epochs):

        # Set the model to training mode
        generator.train()
        discriminator.train()
        r_gen_loss = 0.0
        r_disc_loss = 0.0
        for batch_idx, (l_resized, img_lab_orig) in enumerate(trainloader):
            # Define tensor used for adversarial loss
            target_truth = Variable(Tensor(np.ones((img_lab_orig.size(0), *patch))), requires_grad=False)
            target_false = Variable(Tensor(np.zeros((img_lab_orig.size(0), *patch))), requires_grad=False)
            target_truth = target_truth.to(device)
            target_false = target_false.to(device)

            l_resized = l_resized.to(device)

            # Get the LAB original images and normalize them
            img_lab_orig = lab_normalization.normalize_lab_batch(img_lab_orig)
            img_lab_orig = img_lab_orig.to(device)



            #plot_batch_images(img_lab_orig.detach().to("cpu"), generator.lab_normalization, title="Colored Images")

            # Extract the ground truth ab and convert it to tensor
            ab_groundtruth = img_lab_orig[:, 1:3, :, :]

            # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
            res_ab_groundtruth = resize_to_64x64(ab_groundtruth)
            z_ground, _ = inverse_h_mapping(res_ab_groundtruth, quantized_colorspace)

            # 1. Train the Discriminator
            disc_optimizer.zero_grad()

            with torch.no_grad():
                raw_conv8_output, ab_output = generator(l_resized)
                gen_lab_out = torch.cat((l_resized, ab_output), dim=1)
                gen_lab_out = lab_normalization.normalize_lab_batch(gen_lab_out)

            # Compute the adversarial loss for the discriminator
            disc_ground = discriminator(l_resized, img_lab_orig)
            disc_ground_loss = adv_loss_criterion(disc_ground, target_truth)

            disc_gen = discriminator(l_resized, gen_lab_out)
            disc_gen_loss = adv_loss_criterion(disc_gen, target_false)

            disc_loss = 0.5*(disc_gen_loss + disc_ground_loss)
            #print(f"Discriminator Real Prediction: {disc_ground.mean().item()}, Fake Prediction: {disc_gen.mean().item()}")
            # Check if the loss is above the threshold
            if disc_loss > DISCRIMINATOR_LOSS_THRESHOLD:
                disc_loss.backward()
                disc_optimizer.step()  # Update the discriminator's parameters
            else:
                print(f"Skipping discriminator update. Loss: {disc_loss:.4f}")


            # 2. Train the Generator
            gen_optimizer.zero_grad()

            raw_conv8_output, ab_output = generator(l_resized)
            gen_lab_out = torch.cat((l_resized, ab_output), dim=1)
            gen_lab_out = lab_normalization.normalize_lab_batch(gen_lab_out)
            # Discriminator output for generated images
            with torch.no_grad():
                disc_gen = discriminator(l_resized, gen_lab_out)
            '''
            print("Generator output (ab_output):", ab_output.mean().item(), ab_output.std().item())
            print("Discriminator output (ground):", disc_ground.mean().item(), disc_ground.std().item())
            print("Discriminator output (generated):", disc_gen.mean().item(), disc_gen.std().item())
            '''
            # Generator loss combines adversarial loss, the Z-space loss and the pixel loss
            adv_loss = adv_loss_criterion(disc_gen, target_truth)  # Fool the discriminator
            z_loss = multinomial_cross_entropy_loss_L(raw_conv8_output, z_ground_truth=z_ground)
            pixel_loss = pixel_loss_criterion(gen_lab_out, img_lab_orig)

            print(f"#### Generator Loss Components ####\n-Adv Loss = {adv_loss}\n-Z_Loss = {z_loss}\n-Pixel Loss = {pixel_loss}\n\n")
            gen_loss = adv_loss + PIXEL_FACTOR*pixel_loss

            gen_loss.backward()
            gen_optimizer.step()

            r_gen_loss += gen_loss
            r_disc_loss += disc_loss

            #plot_batch_images(gen_lab_out.detach().to("cpu"), generator.lab_normalization)

            # Cleanup
            del l_resized, img_lab_orig, ab_groundtruth, z_ground, raw_conv8_output, ab_output
            torch.cuda.empty_cache()  # Clear CUDA cache explicitly

        gen_train_loss = r_gen_loss / trainloader.batch_size
        disc_train_loss = r_disc_loss / trainloader.batch_size
        temp_file_generator.write(f"{epoch},{gen_train_loss}\n")
        temp_file_discriminator.write(f"{epoch},{disc_train_loss}\n")

        # Validation
        generator.eval()
        discriminator.eval()
        r_gen_loss = 0.0
        r_disc_loss = 0.0

        with torch.no_grad():
            for l_resized_val, img_lab_orig_val in validloader:
                target_truth = Variable(Tensor(np.ones((img_lab_orig_val.size(0), *patch))), requires_grad=False)
                target_false = Variable(Tensor(np.zeros((img_lab_orig_val.size(0), *patch))), requires_grad=False)

                target_truth = target_truth.to(device)
                target_false = target_false.to(device)

                l_resized_val = l_resized_val.to(device)
                img_lab_orig_val = img_lab_orig_val.to(device)
                img_lab_orig_val = lab_normalization.normalize_lab_batch(img_lab_orig_val)

                img_bw = get_bw_LAB(img_lab_orig_val)

                # Extract the ground truth ab and convert it to tensor
                ab_groundtruth = img_lab_orig_val[:, 1:3, :, :]
                # Normalize the AB groundtruth
                ab_groundtruth = lab_normalization.normalize_ab(ab_groundtruth)

                # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
                ab_groundtruth = resize_to_64x64(ab_groundtruth)
                z_ground, _ = inverse_h_mapping(ab_groundtruth, quantized_colorspace)

                # Calculate the final generated image
                raw_conv8_output, ab_output = generator(l_resized_val)
                gen_lab_out = torch.cat((l_resized_val, ab_output), dim=1)
                gen_lab_out = lab_normalization.normalize_lab_batch(gen_lab_out)


                # Apply discriminator over ground_truth
                disc_ground = discriminator(l_resized_val, img_lab_orig_val)
                disc_ground = disc_ground.to(device)
                disc_ground_loss = adv_loss_criterion(disc_ground, target_truth)

                disc_gen = discriminator(l_resized_val, gen_lab_out)
                disc_gen_loss = adv_loss_criterion(disc_gen, target_false)

                disc_loss = 0.5*(disc_gen_loss + disc_ground_loss)

                r_disc_loss += disc_loss.item()

                # Compute the loss of the generator
                adv_loss = adv_loss_criterion(disc_gen, target_truth)  # Fool the discriminator
                z_loss = multinomial_cross_entropy_loss_L(raw_conv8_output, z_ground_truth=z_ground)
                pixel_loss = pixel_loss_criterion(gen_lab_out, img_lab_orig_val)

                gen_loss = adv_loss + Z_LOSS_FACTOR * z_loss + PIXEL_FACTOR * pixel_loss

                r_gen_loss += gen_loss
                r_disc_loss += disc_loss
                # Cleanup
                del l_resized_val, img_lab_orig_val, ab_groundtruth, z_ground, raw_conv8_output, ab_output
                torch.cuda.empty_cache()

        gen_valid_loss = r_gen_loss / trainloader.batch_size
        disc_valid_loss = r_disc_loss / trainloader.batch_size
        print(f"Epoch [{epoch + 1}/{epochs}], Gen Train Loss: {gen_train_loss:.4f}, Disc Train Loss: {disc_train_loss:.4f};   Gen Valid Loss: {gen_valid_loss:.4f}, Disc Valid Loss: {disc_valid_loss:.4f}")
        #del gen_train_loss, gen_valid_loss, gen_valid_loss, disc_valid_loss

    return temp_file_generator, temp_file_discriminator


def get_bw_LAB(image):
    # Split the LAB tensor into channels
    L = image[:, 0, :, :]  # Lightness channel (B, H, W)
    A = image[:, 1, :, :]  # A channel (B, H, W)
    B = image[:, 2, :, :]  # B channel (B, H, W)

    # Set A and B to neutral values (128)
    A.fill_(0)  # Neutral A channel
    B.fill_(0)  # Neutral B channel

    # Combine the channels back together
    bw_lab_tensor = torch.stack((L, A, B), dim=1)  # (B, C, H, W)
    return bw_lab_tensor
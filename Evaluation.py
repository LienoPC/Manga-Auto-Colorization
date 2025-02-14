import tempfile

import cv2
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
import torcheval
from torch import Tensor
from torch.autograd import Variable
from torchvision.utils import make_grid
from torcheval.metrics import PeakSignalNoiseRatio
from pathlib import Path

from AdversarialNetwork import Discriminator
from ImageDataset import ImageProcess
from Network import resize_to_64x64, inverse_h_mapping, multinomial_cross_entropy_loss_L, ZhangColorizationNetwork, \
    quantized_bins
from torchmetrics import StructuralSimilarityIndexMeasure

from P2PDiscriminator import PatchGAN


def batch_l_channel_to_rgb(l_batch):
    """
    Convert a batch of L-channel images (B, 1, H, W) to RGB (B, 3, H, W),
    assuming neutral A and B channels.

    Args:
        l_batch (torch.Tensor or np.ndarray): Batch of L-channel images with shape (B, 1, H, W)

    Returns:
        np.ndarray: Batch of RGB images with shape (B, 3, H, W), normalized to [0, 1] (float32)
    """
    if isinstance(l_batch, torch.Tensor):
        l_batch = l_batch.cpu().numpy()  # Convert to NumPy if it's a PyTorch tensor

        # Ensure shape (B, 1, H, W) → (B, H, W)
    l_batch = l_batch[:, 0, :, :]

    # Scale L from [0, 100] to [0, 255] for OpenCV compatibility
    l_batch = (l_batch * 255 / 100).astype(np.uint8)

    B, H, W = l_batch.shape

    # Create A and B channels filled with 128 (neutral color)
    a_channel = np.full((B, H, W), 128, dtype=np.uint8)
    b_channel = np.full((B, H, W), 128, dtype=np.uint8)

    # Stack to form LAB images (B, H, W, 3)
    lab_batch = np.stack([l_batch, a_channel, b_channel], axis=-1)

    # Convert LAB to RGB for each image in the batch
    rgb_batch = np.array([cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB) for lab_img in lab_batch])

    # Convert to PyTorch format: (B, H, W, 3) → (B, 3, H, W)
    rgb_batch = np.transpose(rgb_batch, (0, 3, 1, 2))

    # Normalize to [0, 1] for deep learning (optional)
    rgb_batch = rgb_batch.astype(np.float32) / 255.0

    return torch.tensor(rgb_batch)  # Return as PyTorch tenso

def lab_to_rgb_batch(lab_tensors):
    """Convert a batch of LAB tensors (BATCH, 3, HEIGHT, WIDTH) to RGB tensors with the same shape."""
    batch_size, _, height, width = lab_tensors.shape
    rgb_tensors = torch.zeros((batch_size, 3, height, width), dtype=torch.float32)

    for i, lab_tensor in enumerate(lab_tensors):
        lab_numpy = lab_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)

        # Clamp values to valid range before conversion
        lab_numpy = np.clip(lab_numpy, [0, -128, -128], [100, 127, 127]).astype(np.float32)

        rgb_image = cv2.cvtColor(lab_numpy, cv2.COLOR_LAB2RGB)
        #print(f"Rgb image: {rgb_image}")

        rgb_tensors[i] = torch.from_numpy(rgb_image).permute(2, 0, 1).to(dtype=torch.float32)
        #print(f"Rgb tensor: {rgb_tensors[i]}")

    return rgb_tensors

'''
def lab_to_rgb_batch(l_resized, ab_channel):
    """Convert a batch of LAB tensors (BATCH, 3, HEIGHT, WIDTH) to RGB tensors with the same shape."""
    batch_size, _, height, width =  ab_channel.shape
    rgb_tensors = torch.zeros((batch_size, 3, height, width), dtype=torch.uint8)

    for i, (l_channel, ab_c) in enumerate(zip(l_resized, ab_channel)):
        print(f"Batch {i}, l_channel {l_channel.shape}, ab_channel {ab_c.shape}")
        #rgb_tensors[i] = torch.tensor(ImageProcess.postprocess_tens(l_channel, ab_c)).permute(0, 3, 1, 2)
        rgb_tensors[i] = torch.from_numpy(ImageProcess.postprocess_tens(l_channel, ab_c)).permute(2, 0, 1).to(dtype=torch.uint8)
    return rgb_tensors
'''



def save_rgb_batch_as_jpg(rgb_tensors, output_folder, prefix, batch):
    """Save a batch of unnormalized RGB tensors (BATCH, 3, HEIGHT, WIDTH) as JPG files."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, rgb_tensor in enumerate(rgb_tensors):
        rgb_image = (rgb_tensor.permute(1, 2, 0).cpu().numpy()*255.0).astype(np.uint8) # Convert to (H, W, 3)
        filename = output_folder / f"{prefix}_{i}_{batch}.jpg"
        cv2.imwrite(str(filename), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(rgb_tensors)} images to {output_folder}")

def test_zhang(device, test_loader, lab_normalization, img_dim, model_path):
    save_dir = f"./SavedTests/Zhang/ColoredImages"

    model = ZhangColorizationNetwork(lab_normalization)
    checkpoint = torch.load(model_path, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    temp_file = tempfile.NamedTemporaryFile(mode="w+")

    r_gen_loss = 0.0
    r_disc_loss = 0.0
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)

    quantized_colorspace = quantized_bins().to(device)
    # Validation
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    batch = 0

    with torch.no_grad():
        for l_resized, img_lab_orig in test_loader:
            l_resized = l_resized.to(device)
            img_lab_orig = img_lab_orig.to(device)

            l_rgb = batch_l_channel_to_rgb(l_resized)
            save_rgb_batch_as_jpg(l_rgb, f"{save_dir}/Input", 'patch', batch)
            # Extract the ground truth ab and convert it to tensor
            ab_groundtruth_val = img_lab_orig[:, 1:3, :, :]
            # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
            ab_groundtruth_val = resize_to_64x64(ab_groundtruth_val)
            z_ground_val, _ = inverse_h_mapping(ab_groundtruth_val, quantized_colorspace)

            # Apply the model
            raw_conv8_output_val, ab_output = model(l_resized)
            gen_lab_out = torch.cat((l_resized, ab_output), dim=1)

            z_loss = multinomial_cross_entropy_loss_L(raw_network_output=raw_conv8_output_val,
                                                      z_ground_truth=z_ground_val)

            running_loss += float(z_loss.item())
            print(f"Running loss: {running_loss}\n")
            # After loss we compute PSNR
            psnr_metric.update(gen_lab_out, img_lab_orig)
            ssim_metric.update(gen_lab_out, img_lab_orig)

            gen_lab_out = lab_to_rgb_batch(gen_lab_out)
            img_lab_orig = lab_to_rgb_batch(img_lab_orig)
            save_rgb_batch_as_jpg(gen_lab_out, save_dir, 'patch', batch)
            save_rgb_batch_as_jpg(img_lab_orig, f"{save_dir}/Original", 'patch', batch)



            batch += 1

    test_loss = running_loss / len(test_loader)
    psnr_loss = psnr_metric.compute()
    ssim_loss = ssim_metric.compute()
    temp_file.write(f"Test Generator Loss: {test_loss}\n")
    temp_file.write(f"PSNR: {psnr_loss}\n")
    temp_file.write(f"SSIM: {ssim_loss}\n")
    temp_file.flush()
    return temp_file


def test_patch(device, test_loader, lab_normalization, img_dim, gen_path, disc_path):
    save_dir = f"./SavedTests/Patch/ColoredImages"

    generator = ZhangColorizationNetwork(lab_normalization)
    checkpoint = torch.load(gen_path, weights_only=True)

    generator.load_state_dict(checkpoint['model_state_dict'])

    discriminator = PatchGAN(3)
    checkpoint = torch.load(disc_path, weights_only=True)
    discriminator.load_state_dict(checkpoint['model_state_dict'])

    generator.to(device)
    discriminator.to(device)

    generator.eval()  # model to eval
    discriminator.eval()

    temp_file_generator = tempfile.NamedTemporaryFile(mode="w+")
    temp_file_discriminator = tempfile.NamedTemporaryFile(mode="w+")

    r_gen_loss = 0.0
    r_disc_loss = 0.0
    # Define loss criterion
    adv_loss_criterion = nn.BCELoss()
    pixel_loss_criterion = nn.L1Loss()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)

    quantized_colorspace = quantized_bins().to(device)
    # Loss multiply factors
    DISCRIMINATOR_LOSS_THRESHOLD = 0.4
    PIXEL_FACTOR = 1.0
    Z_LOSS_FACTOR = 1.0

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, img_dim // 2 ** 4, img_dim // 2 ** 4)
    batch = 0
    with torch.no_grad():
        for l_resized, img_lab_orig in test_loader:
            target_truth = Variable(Tensor(np.ones((img_lab_orig.size(0), *patch))), requires_grad=False)
            target_false = Variable(Tensor(np.zeros((img_lab_orig.size(0), *patch))), requires_grad=False)

            target_truth = target_truth.to(device)
            target_false = target_false.to(device)

            l_resized = l_resized.to(device)
            img_lab_orig = img_lab_orig.to(device)
            img_lab_orig = lab_normalization.normalize_lab_batch(img_lab_orig)

            # Extract the ground truth ab and convert it to tensor
            ab_groundtruth = img_lab_orig[:, 1:3, :, :]

            # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
            ab_groundtruth = resize_to_64x64(ab_groundtruth)
            z_ground, _ = inverse_h_mapping(ab_groundtruth, quantized_colorspace)

            # Calculate the final generated image
            raw_conv8_output, ab_output = generator(l_resized)
            gen_lab_out = torch.cat((l_resized, ab_output), dim=1)
            gen_lab_out = lab_normalization.normalize_lab_batch(gen_lab_out)

            # Apply discriminator over ground_truth
            disc_ground = discriminator(l_resized, img_lab_orig)
            disc_ground = disc_ground.to(device)
            disc_ground_loss = adv_loss_criterion(disc_ground, target_truth)

            disc_gen = discriminator(l_resized, gen_lab_out)
            disc_gen_loss = adv_loss_criterion(disc_gen, target_false)

            disc_loss = 0.5 * (disc_gen_loss + disc_ground_loss)

            r_disc_loss += disc_loss.item()

            # Compute the loss of the generator
            adv_loss = adv_loss_criterion(disc_gen, target_truth)  # Fool the discriminator
            z_loss = multinomial_cross_entropy_loss_L(raw_conv8_output, z_ground_truth=z_ground)
            pixel_loss = pixel_loss_criterion(gen_lab_out, img_lab_orig)

            gen_loss = adv_loss + Z_LOSS_FACTOR * z_loss + PIXEL_FACTOR * pixel_loss
            # plot_batch_images(gen_lab_out.detach().to("cpu"), generator.lab_normalization)

            gen_lab_out = lab_normalization.unnormalize_lab_batch(gen_lab_out)
            img_lab_orig = lab_normalization.unnormalize_lab_batch(img_lab_orig)
            # After loss we compute PSNR
            psnr_metric.update(gen_lab_out, img_lab_orig)
            ssim_metric.update(gen_lab_out, img_lab_orig)
            gen_lab_out = lab_to_rgb_batch(gen_lab_out)
            img_lab_orig = lab_to_rgb_batch(img_lab_orig)
            save_rgb_batch_as_jpg(gen_lab_out, save_dir, 'patch', batch)
            #save_rgb_batch_as_jpg(gen_lab_out, f"{save_dir}/Original", 'patch', batch)



            r_gen_loss += gen_loss
            r_disc_loss += disc_loss
            # Cleanup
            del l_resized, img_lab_orig, ab_groundtruth, z_ground, raw_conv8_output, ab_output
            #torch.cuda.empty_cache()
            batch += 1

    gen_test_loss = r_gen_loss / len(test_loader)
    disc_test_loss = r_disc_loss / len(test_loader)
    psnr_loss = psnr_metric.compute()
    ssim_loss = ssim_metric.compute()
    temp_file_generator.write(f"Test Generator Loss: {gen_test_loss}\n")
    temp_file_discriminator.write(f"Test Discriminator Loss:{disc_test_loss}\n")
    temp_file_generator.write(f"PSNR: {psnr_loss}\n")
    temp_file_generator.write(f"SSIM: {ssim_loss}\n")
    temp_file_generator.flush()
    temp_file_discriminator.flush()
    return temp_file_generator, temp_file_discriminator



def test_adv(device, test_loader, lab_normalization, img_dim, gen_path, disc_path):
    save_dir = f"./SavedTests/Adv/ColoredImages"

    DISCRIMINATOR_LOSS_THRESHOLD = 1.0
    PIXEL_FACTOR = 1.0
    Z_LOSS_FACTOR = 1.0
    # Compute the quantized bins and move them to the correct device
    adv_loss_criterion = nn.BCELoss()
    pixel_loss_criterion = nn.MSELoss()
    generator = ZhangColorizationNetwork(lab_normalization)
    checkpoint = torch.load(gen_path, weights_only=True)

    generator.load_state_dict(checkpoint['model_state_dict'])

    discriminator = Discriminator()
    checkpoint = torch.load(disc_path, weights_only=True)
    discriminator.load_state_dict(checkpoint['model_state_dict'])

    temp_file_generator = tempfile.NamedTemporaryFile(mode="w+")
    temp_file_discriminator = tempfile.NamedTemporaryFile(mode="w+")

    generator.to(device)
    discriminator.to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)
    # Validation
    generator.eval()
    discriminator.eval()
    r_gen_loss = 0.0
    r_disc_loss = 0.0
    quantized_colorspace = quantized_bins().to(device)
    batch = 0
    with torch.no_grad():
        for l_resized, img_lab_orig in test_loader:
            target_truth = Variable(torch.ones(l_resized.shape[0], 1), requires_grad=False)
            target_false = Variable(torch.zeros(l_resized.shape[0], 1), requires_grad=False)

            target_truth = target_truth.to(device)
            target_false = target_false.to(device)

            l_resized = l_resized.to(device)
            img_lab_orig = img_lab_orig.to(device)
            img_lab_orig = lab_normalization.normalize_lab_batch(img_lab_orig)
            # Extract the ground truth ab and convert it to tensor
            ab_groundtruth = img_lab_orig[:, 1:3, :, :]
            # Normalize the AB groundtruth
            ab_groundtruth = lab_normalization.normalize_ab(ab_groundtruth)

            # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
            ab_groundtruth = resize_to_64x64(ab_groundtruth)
            z_ground, _ = inverse_h_mapping(ab_groundtruth, quantized_colorspace)

            # Calculate the final generated image
            raw_conv8_output, ab_output = generator(l_resized)
            gen_lab_out = torch.cat((l_resized, ab_output), dim=1)
            gen_lab_out = lab_normalization.normalize_lab_batch(gen_lab_out)

            # Apply discriminator over ground_truth
            disc_ground = discriminator(img_lab_orig)
            disc_ground = disc_ground.to(device)
            disc_ground_loss = adv_loss_criterion(disc_ground, target_truth)

            disc_gen = discriminator(gen_lab_out)
            disc_gen_loss = adv_loss_criterion(disc_gen, target_false)

            disc_loss = disc_gen_loss + disc_ground_loss

            r_disc_loss += disc_loss.item()

            # Compute the loss of the generator
            adv_loss = adv_loss_criterion(disc_gen, target_truth)  # Fool the discriminator
            z_loss = multinomial_cross_entropy_loss_L(raw_conv8_output, z_ground_truth=z_ground)
            pixel_loss = pixel_loss_criterion(gen_lab_out, img_lab_orig)

            gen_loss = adv_loss + Z_LOSS_FACTOR * z_loss + PIXEL_FACTOR * pixel_loss

            gen_lab_out = lab_normalization.unnormalize_lab_batch(gen_lab_out)
            img_lab_orig = lab_normalization.unnormalize_lab_batch(img_lab_orig)
            # After loss we compute PSNR
            psnr_metric.update(gen_lab_out, img_lab_orig)
            ssim_metric.update(gen_lab_out, img_lab_orig)
            gen_lab_out = lab_to_rgb_batch(gen_lab_out)
            img_lab_orig = lab_to_rgb_batch(img_lab_orig)
            save_rgb_batch_as_jpg(gen_lab_out, save_dir, 'patch', batch)
            #save_rgb_batch_as_jpg(gen_lab_out, f"{save_dir}/Original", 'patch', batch)



            r_gen_loss += gen_loss.item()
            r_disc_loss += disc_loss.item()
            # Cleanup
            del l_resized, img_lab_orig, ab_groundtruth, z_ground, raw_conv8_output, ab_output, target_truth, target_false
            #torch.cuda.empty_cache()
            batch += 1

    gen_test_loss = r_gen_loss / len(test_loader)
    disc_test_loss = r_disc_loss / len(test_loader)
    psnr_loss = psnr_metric.compute()
    ssim_loss = ssim_metric.compute()
    temp_file_generator.write(f"Test Generator Loss: {gen_test_loss}\n")
    temp_file_discriminator.write(f"Test Discriminator Loss:{disc_test_loss}\n")
    temp_file_generator.write(f"PSNR: {psnr_loss}\n")
    temp_file_generator.write(f"SSIM: {ssim_loss}\n")
    temp_file_generator.flush()
    temp_file_discriminator.flush()
    return temp_file_generator, temp_file_discriminator



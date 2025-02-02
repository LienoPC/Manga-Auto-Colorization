import tempfile

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
from Network import resize_to_64x64, inverse_h_mapping, multinomial_cross_entropy_loss_L
from torchmetrics import StructuralSimilarityIndexMeasure


def test_patch(generator, discriminator, device, test_loader, lab_normalization, temp_file_generator, temp_file_discriminator, quantized_colorspace, img_dim):
    generator.eval()  # model to eval
    test_loss = 0
    correct = 0

    r_gen_loss = 0.0
    r_disc_loss = 0.0
    # Define loss criterion
    adv_loss_criterion = nn.BCELoss()
    pixel_loss_criterion = nn.L1Loss()
    psnr_metric = PeakSignalNoiseRatio()
    ssim_metric = StructuralSimilarityIndexMeasure(reduction='elementwise_mean')
    # Loss multiply factors
    DISCRIMINATOR_LOSS_THRESHOLD = 0.4
    PIXEL_FACTOR = 1.0
    Z_LOSS_FACTOR = 3.0

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, img_dim // 2 ** 4, img_dim // 2 ** 4)


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

        # After loss we compute PSNR
        psnr_metric.update(gen_lab_out, img_lab_orig)
        ssim_metric.update(gen_lab_out, img_lab_orig)
        r_gen_loss += gen_loss
        r_disc_loss += disc_loss
        # Cleanup
        del l_resized, img_lab_orig, ab_groundtruth, z_ground, raw_conv8_output, ab_output
        torch.cuda.empty_cache()

    gen_valid_loss = r_gen_loss / len(test_loader)
    disc_valid_loss = r_disc_loss / len(test_loader)
    psnr_loss = psnr_metric.compute()
    ssim_loss = ssim_metric.compute()
    temp_file_generator.write(f"Test Generator Loss: {gen_valid_loss}\n")
    temp_file_discriminator.write(f"Test Discriminator Loss:{disc_valid_loss}\n")
    temp_file_generator.write(f"PSNR: {psnr_loss}")
    temp_file_generator.write(f"SSIM: {ssim_loss}")
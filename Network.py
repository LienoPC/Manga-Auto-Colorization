import math

import torchvision
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage import color
from PIL import Image
import torch.nn.functional as F

class ZhangColorizationNetwork(nn.Module):
    def __init__(self):
        super(ZhangColorizationNetwork,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(256))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512))

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512))

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512))

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, padding=1, stride=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=313, kernel_size=1, padding=0, stride=1, bias=True)
            )

        self.ab = nn.Sequential(
            nn.Softmax(dim=1),
            nn.Conv2d(in_channels=313, out_channels=2, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Upsample(scale_factor=4, mode='bilinear'))

    def forward(self, image):
        '''
            Starting from the L channel extracted from the colored image (during training) or from the L channel
            of the bw image, feed the network
        '''
        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)

        ab_channel = self.ab(conv8)

        return conv8, ab_channel

    @staticmethod
    def res_image(image, new_size=(256, 256), resample=3):
        return np.asarray(Image.fromarray(image).resize((new_size[1], new_size[0]), resample=resample))

    @staticmethod
    def preprocess_img(img_rgb_orig, new_size=(256, 256), resample=3):

        img_rgb_rs = ZhangColorizationNetwork.res_image(img_rgb_orig, new_size=new_size, resample=resample)

        img_lab_orig = color.rgb2lab(img_rgb_orig)
        img_lab_rs = color.rgb2lab(img_rgb_orig)

        img_l_orig = img_lab_orig[:, :, 0]
        img_l_rs = img_lab_rs[:, :, 0]

        tens_orig_l = torch.Tensor(img_l_orig)[None, :, :]
        tens_rs_l = torch.Tensor(img_l_rs)[None, :, :]
        return tens_orig_l, tens_rs_l


    @staticmethod
    def postprocess_tens(tens_orig_l, out_ab):
        # tens_orig_l 	1 x 1 x H_orig x W_orig
        # out_ab 		1 x 2 x H x W

        size_orig = tens_orig_l.shape[2:]
        size = out_ab.shape[2:]

        # call resize function if needed
        if size_orig[0] != size[0] or size_orig[1] != size[1]:
            out_ab_orig = F.interpolate(out_ab, size=size_orig, mode='bilinear')
        else:
            out_ab_orig = out_ab

        out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
        return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))


    def point_estimate_H(self,Z_predicted):
        T = 1  # defines how the temperature of the image is re-adjusted (1 -> no changes)

        # First compute fT(z)
        num = torch.exp(torch.log(Z_predicted) / T)
        den = 0
        for q in range(Z_predicted.size(2)):
            channel = Z_predicted[:, :, q]
            den += torch.exp(torch.log(channel) / T)

        return torch.mean(num / den, dim=(0, 1))

def zhang_train(model, trainloader, validloader, device, optimizer, epochs=2):
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

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for l_resized, img_lab_orig in trainloader:

            # Get the LAB original image
            print(img_lab_orig.shape)
            optimizer.zero_grad()
            print(l_resized.shape)
            l_resized = l_resized.to(device)

            # Extract the ground truth ab and convert it to tensor
            ab_groundtruth = img_lab_orig[:,1:3,:,:]
            # Apply soft-encoding (using nearest neighbor) to map Yab to Zab
            Z_ground = inverse_H_mapping(ab_groundtruth)

            Z_predicted, ab_output = model(l_resized) # Z_predicted
            # Compute the custom loss over the Z space
            loss = multinomial_cross_entropy_loss_L(Z_predicted=Z_predicted, Z_ground_truth=Z_ground) # TO DEFINE
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)

        # Validation
        model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():
            for image in validloader:
                image = image.to(device)
                ab_output = model(image)
                ab_groundtruth = LabNormalization.get_ab_channel(image)
                loss = multinomial_cross_entropy_loss_L(Z_predicted=ab_output, Z_ground_truth=ab_groundtruth) # TO DEFINE
                running_loss += loss.item()

        valid_loss = running_loss / len(validloader)
        valid_losses.append(valid_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")


# Custom loss function
def multinomial_cross_entropy_loss_L(Z_predicted, Z_ground_truth):
    loss = torch.zeroes()
    for h, w in range(Z_ground_truth.size(0,1)):
        for q in range(Z_ground_truth.size(2)):
            loss += Z_ground_truth[h,w,q] * torch.log(Z_predicted[h,w,q])

    loss = - loss

    return loss


# To obtain Z_ground_truth we define the relative mapping H^(-1)
def gaussian_weight(distances, sigma):
    return torch.exp(-distances ** 2 / (2 * sigma ** 2))


def inverse_H_mapping(ab_channels, sigma=5, chunk_size=200):
    """
    GPU-based inverse H mapping using Gaussian-weighted 5-NN.
    """
    B, C, H, W = ab_channels.shape  # Assuming shape is [Batch, 2, Height, Width]
    print(f"Batch: {B}, Channels: {C}, Height: {H}, Width: {W}")

    # Flatten spatial dimensions
    ab_flat = ab_channels.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)  # [N, 2]
    N = ab_flat.shape[0]  # Total number of pixels across the batch

    # Initialize the output tensor
    output_flat = torch.zeros_like(ab_flat, device=ab_channels.device)

    # Process in chunks to save memory
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = ab_flat[start:end]  # Current chunk [chunk_size, 2]

        # Compute distances between the chunk and all pixels
        distances = torch.cdist(chunk, ab_flat, p=2)  # [chunk_size, N]

        # Find 5 nearest neighbors for each pixel in the chunk
        knn_distances, knn_indices = torch.topk(distances, k=5, largest=False, dim=1)  # [chunk_size, 5]

        # Gather neighbors and compute weighted sums
        neighbors = ab_flat[knn_indices]  # [chunk_size, 5, 2]
        weights = gaussian_weight(knn_distances, sigma)  # [chunk_size, 5]
        weighted_sum = torch.sum(neighbors * weights.unsqueeze(-1), dim=1)  # [chunk_size, 2]

        # Normalize weights
        output_flat[start:end] = weighted_sum / weights.sum(dim=1, keepdim=True)

    # Reshape back to original shape
    output = output_flat.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
    return output

class LabNormalization():

    def __init__(self):
        self.l_channel_norm = 100
        self.ab_channel_norm = 110

    @staticmethod
    def get_l_channel(x):
        l_channel = x[:, :, 0].float()
        l_channel = l_channel / 255.0
        l_channel = l_channel.unsqueeze(0).unsqueeze(0)
        return l_channel

    @staticmethod
    def get_ab_channel(x):
        l_channel = x[:, :, 0].float()
        l_channel = l_channel / 255.0
        l_channel = l_channel.unsqueeze(0).unsqueeze(0)
        return l_channel


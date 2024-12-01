import math

import torchvision
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision import models, transforms, datasets
from torchsummary import summary

from ImageDataset import LabNormalization, ImageDataset


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
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(512))

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1, dilation=2, bias=True),
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
        # Add here conversion to LAB color scheme
        original_l_channel, resized_l_channel = ImageDataset.preprocess_img(image)

        conv1 = self.conv1(resized_l_channel) #here should go a normalization for L channel
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)

        ab_channel = self.ab(conv8)

        return ab_channel



def zhang_train(model, trainloader, validloader, device, optimizer, epochs=10):
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
        for resized_l_channel, original_l_channel, image in trainloader:
            resized_l_channel = resized_l_channel.to(device)
            optimizer.zero_grad()

            ab_groundtruth = LabNormalization.get_ab_channel(image)
            ab_output = model(resized_l_channel) # Z_predicted

            # Map the ab value of the ground truth using Z function

            loss = criterion(outputs, labels) # TO DEFINE
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)

        # Validation
        model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        valid_loss = running_loss / len(validloader)
        valid_losses.append(valid_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")



#def multinomial_cross_entropy_loss_L(Z_predicted, Z_ground_truth):




# To obtain Z_ground_truth I must define H function and the relative H^(-1)
def point_estimate_H(Z_predicted):
    T = 1 # defines how the temperature of the image is re-adjusted (1 -> no changes)

    # First compute fT(z)
    num = torch.exp(torch.log(Z_predicted)/T)
    den = 0
    for q in range(Z_predicted.size(2)):
        channel = Z_predicted[:, :, q]
        den += torch.exp(torch.log(channel) / T)

    return torch.mean(num/den, dim=(0,1))


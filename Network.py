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
from torchvision.utils import make_grid

from ImageDataset import ImageProcess


def plot_batch_images(batch, lab_normalization, title="Batch Training Images", nrow=8, figsize=(12, 12), normalize=True):
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


class ZhangColorizationNetwork(nn.Module):
    def __init__(self, lab_normalization):
        super(ZhangColorizationNetwork, self).__init__()
        self.lab_normalization = lab_normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2, bias=True),
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
        conv1 = self.conv1(self.lab_normalization.normalize_l(image))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)

        ab_channel = self.lab_normalization.unnormalize_ab(self.ab(conv8))
        #ab_channel = self.lab_normalization.unnormalize_ab(point_estimate_H(ab_channel))
        return conv8,ab_channel

    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def zhang_train_step(model, trainloader, device, optimizer, lab_normalization, temp_file, quantized_colorspace, epoch):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    pixel_loss_criterion = nn.L1Loss()

    for batch_idx, (l_resized, img_lab_orig) in enumerate(trainloader):
        # Get the LAB original image
        optimizer.zero_grad()
        l_resized = l_resized.to(device)
        img_lab_orig = img_lab_orig.to(device)

        # Extract the ground truth ab and convert it to tensor
        ab_groundtruth = img_lab_orig[:, 1:3, :, :]
        # Normalize the AB groundtruth
        ab_groundtruth = lab_normalization.normalize_ab(ab_groundtruth)

        # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
        ab_groundtruth = resize_to_64x64(ab_groundtruth)
        z_ground, _ = inverse_h_mapping(ab_groundtruth, quantized_colorspace)

        # Apply the model
        raw_conv8_output, ab_output = model(l_resized)
        '''
        print("\n\nZ Space printed:\n ")
        print(f"Predicted Z: \n shape:")
        print(raw_conv8_output.shape)
        print("\n\n")
        print(raw_conv8_output)
        print(f"Ground Z: \n  shape:")
        print(z_ground.shape)
        print("\n\n")
        print(z_ground)
        '''
        gen_lab_out = torch.cat((l_resized, ab_output), dim=1)
        gen_lab_out = lab_normalization.normalize_lab_batch(gen_lab_out)
        # Compute the custom loss over the Z space
        z_loss = multinomial_cross_entropy_loss_L(raw_network_output=raw_conv8_output, z_ground_truth=z_ground)
        pixel_loss = pixel_loss_criterion(gen_lab_out, img_lab_orig)

        loss = z_loss + pixel_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # plot_batch_images(gen_lab_out.detach().to("cpu"), model.lab_normalization)

    train_loss = running_loss / len(trainloader)
    temp_file.write(f"{epoch},{train_loss}\n")

    return temp_file, train_loss


def zhang_valid_step(model, validloader, device, optimizer, lab_normalization, temp_file, quantized_colorspace, epoch):
    # Validation
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():
        for l_resized_val, img_lab_orig_val in validloader:
            l_resized_val = l_resized_val.to(device)
            img_lab_orig_val = img_lab_orig_val.to(device)

            # Extract the ground truth ab and convert it to tensor
            ab_groundtruth_val = img_lab_orig_val[:, 1:3, :, :]
            # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
            ab_groundtruth_val = resize_to_64x64(ab_groundtruth_val)
            z_ground_val, _ = inverse_h_mapping(ab_groundtruth_val, quantized_colorspace)

            # Apply the model
            raw_conv8_output_val, ab_output = model(l_resized_val)
            loss = multinomial_cross_entropy_loss_L(raw_network_output=raw_conv8_output_val,
                                                    z_ground_truth=z_ground_val)

            running_loss += loss.item()

    valid_loss = running_loss / len(validloader)
    temp_file.write(f"{epoch},{valid_loss}\n")

    return temp_file, valid_loss

def DEPRECATED_zhang_train(model, trainloader, validloader, device, optimizer, lab_normalization, epochs=2):
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
    pixel_loss_criterion = nn.BCELoss()
    print("Started training...")
    temp_file_train = tempfile.NamedTemporaryFile(mode="w+")
    temp_file_valid = tempfile.NamedTemporaryFile(mode="w+")
    for epoch in range(epochs):

        model.train()  # Set the model to training mode
        running_loss = 0.0
        for batch_idx, (l_resized, img_lab_orig) in enumerate(trainloader):
            # Get the LAB original image
            optimizer.zero_grad()
            l_resized = l_resized.to(device)
            img_lab_orig = img_lab_orig.to(device)

            # Extract the ground truth ab and convert it to tensor
            ab_groundtruth = img_lab_orig[:, 1:3, :, :]
            # Normalize the AB groundtruth
            ab_groundtruth = lab_normalization.normalize_ab(ab_groundtruth)

            # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
            ab_groundtruth = resize_to_64x64(ab_groundtruth)
            z_ground, _ = inverse_h_mapping(ab_groundtruth, quantized_colorspace)

            # Apply the model
            raw_conv8_output, ab_output = model(l_resized)

            # Compute the custom loss over the Z space
            '''
            print("\n\nZ Space printed:\n ")
            print(f"Predicted Z: \n shape:")
            print(raw_conv8_output.shape)
            print("\n\n")
            print(raw_conv8_output)
            print(f"Ground Z: \n  shape:")
            print(z_ground.shape)
            print("\n\n")
            print(z_ground)
            '''
            gen_lab_out = torch.cat((l_resized, ab_output), dim=1)
            gen_lab_out = lab_normalization.normalize_lab_batch(gen_lab_out)
            z_loss = multinomial_cross_entropy_loss_L(raw_network_output=raw_conv8_output, z_ground_truth=z_ground)
            pixel_loss = pixel_loss_criterion(gen_lab_out, img_lab_orig)

            loss = z_loss + pixel_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{round(len(trainloader)/trainloader.batch_size)}], Loss: {loss.item()}")

            #plot_batch_images(gen_lab_out.detach().to("cpu"), model.lab_normalization)


        train_loss = running_loss/len(trainloader)
        temp_file_train.write(f"{epoch},{train_loss}\n")
        # Validation
        model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        with torch.no_grad():
            for l_resized_val, img_lab_orig_val in validloader:
                l_resized_val = l_resized_val.to(device)
                img_lab_orig_val = img_lab_orig_val.to(device)

                # Extract the ground truth ab and convert it to tensor
                ab_groundtruth_val = img_lab_orig_val[:, 1:3, :, :]
                # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
                ab_groundtruth_val = resize_to_64x64(ab_groundtruth_val)
                z_ground_val, _ = inverse_h_mapping(ab_groundtruth_val, quantized_colorspace)

                # Apply the model
                raw_conv8_output_val, ab_output = model(l_resized_val)
                loss = multinomial_cross_entropy_loss_L(raw_network_output=raw_conv8_output_val, z_ground_truth=z_ground_val)

                running_loss += loss.item()

        valid_loss = running_loss / len(validloader)
        temp_file_train.write(f"{epoch},{valid_loss}\n")
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    return temp_file_train, temp_file_train



def zhang_train_progressed(model, trainloader, validloader, device, optimizer, epochs=2):
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

    learning_train_loss = []
    learning_valid_loss = []



    # Compute the quantized bins and move them to the correct device
    quantized_colorspace = quantized_bins().to(device)

    print("Started training...")
    for i in range(1,2):
        # Subdivide dataset in order to plot the learning curve
        train_sub = torch.utils.data.Subset(trainloader.dataset, range(0, i*300))
        valid_sub = torch.utils.data.Subset(validloader.dataset, range(0, i*80))

        train_sub_loader = torch.utils.data.DataLoader(train_sub, batch_size=trainloader.batch_size, shuffle=True)
        valid_sub_loader = torch.utils.data.DataLoader(valid_sub, batch_size=validloader.batch_size, shuffle=True)
        train_losses = []
        valid_losses = []
        for epoch in range(epochs):

            model.train()  # Set the model to training mode
            running_loss = 0.0
            for batch_idx, (l_resized, img_lab_orig) in train_sub_loader:
                # Get the LAB original image


                optimizer.zero_grad()

                l_resized = l_resized.to(device)
                img_lab_orig = img_lab_orig.to(device)

                # Extract the ground truth ab and convert it to tensor
                ab_groundtruth = img_lab_orig[:, 1:3, :, :]

                # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
                ab_groundtruth = resize_to_64x64(ab_groundtruth)

                z_ground, _ = inverse_h_mapping(ab_groundtruth, quantized_colorspace)

                # Apply the model
                raw_conv8_output, ab_output = model(l_resized)

                # Compute the custom loss over the Z space
                loss = multinomial_cross_entropy_loss_L(raw_network_output=raw_conv8_output, z_ground_truth=z_ground)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss
            train_losses.append(train_loss)

            # Validation
            model.eval()  # Set the model to evaluation mode
            running_loss = 0.0
            with torch.no_grad():
                for l_resized_val, img_lab_orig_val in valid_sub_loader:
                    l_resized_val = l_resized_val.to(device)
                    img_lab_orig_val = img_lab_orig_val.to(device)

                    # Extract the ground truth ab and convert it to tensor
                    ab_groundtruth_val = img_lab_orig_val[:, 1:3, :, :]

                    # Resize the ground truth and apply soft-encoding (using nearest neighbor) to map Yab to Zab
                    ab_groundtruth_val = resize_to_64x64(ab_groundtruth_val)
                    z_ground_val, _ = inverse_h_mapping(ab_groundtruth_val, quantized_colorspace)

                    # Apply the model
                    raw_conv8_output_val, ab_output = model(l_resized_val)

                    loss = multinomial_cross_entropy_loss_L(raw_network_output=raw_conv8_output_val, z_ground_truth=z_ground_val)

                    running_loss += loss.item()

            valid_loss = running_loss
            valid_losses.append(valid_loss)
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        learning_train_loss.append(train_losses.pop())
        learning_valid_loss.append(valid_losses.pop())
        model.reset_weights()
        del(train_sub)
        del(valid_sub)
        del(train_sub_loader)
        del(valid_sub_loader)

    return learning_train_loss, learning_valid_loss

# Custom loss function
def multinomial_cross_entropy_loss_L(raw_network_output, z_ground_truth):
    # Apply softmax to ensure Z_predicted is a valid probability distribution
    z_predicted = torch.softmax(raw_network_output, dim=-1)

    # Masking to avoid getting a nan result from applying a log function to the z_predicted values
    #mask = z_predicted > 0
    #masked_log = torch.zeros_like(z_predicted, device=z_predicted.device)
    #masked_log[mask] = torch.log(z_predicted[mask])

    eps = 1e-8
    log_z_predicted = torch.log(z_predicted + eps)
    # print(f'Z_ground_truth shape: {Z_ground_truth.shape}')
    # print(f'Z_predicted shape: {Z_predicted.shape}')



    loss = -torch.sum(z_ground_truth * log_z_predicted)
    # Normalize loss by the number of elements
    loss /= z_ground_truth.numel()
    #print(f'Loss: {loss}')
    del log_z_predicted, z_predicted, eps
    return loss


def quantized_bins(grid_step=10, valid_range_a=(-110, 110), valid_range_b=(-110, 110), num_bins=313):
    """
    Generates a grid of quantized bins for the 'a' and 'b' components in the CIELAB color space.
    The grid is defined with a step size for each dimension (a, b), and only in-gamut values are selected
    to obtain a list of 313 valid bins.

    Parameters:
        grid_step (int): Step size for each dimension (a, b).
        valid_range_a (tuple): Valid range for the 'a' component (default (-110, 110)).
        valid_range_b (tuple): Valid range for the 'b' component (default (-110, 110)).
        num_bins (int): Number of quantized values to keep (default 313, as specified in the paper).

    Returns:
        torch.Tensor: A tensor containing the 313 quantized bins in the ab color space.
    """

    # Generate a grid for a and b with step size grid_step
    a_values = np.arange(valid_range_a[0], valid_range_a[1] + grid_step, grid_step)
    b_values = np.arange(valid_range_b[0], valid_range_b[1] + grid_step, grid_step)

    # Create all possible combinations of (a, b)
    ab_grid = np.array(np.meshgrid(a_values, b_values)).T.reshape(-1, 2)  # Shape: (n_points, 2)

    # Check if values are in gamut (convert LAB to sRGB and check bounds)
    L_fixed = 50  # Common choice for a fixed luminance
    lab_colors = np.hstack([np.full(shape=(ab_grid.shape[0], 1), fill_value=L_fixed), ab_grid])  # Add L column
    rgb_colors = lab2rgb(lab_colors.reshape(-1, 1, 3)).reshape(-1, 3)  # Convert LAB to RGB
    in_gamut_mask = np.all((rgb_colors >= 0) & (rgb_colors <= 1), axis=1)  # RGB in range [0, 1]

    # Keep only in-gamut values
    valid_ab_values = ab_grid[in_gamut_mask]

    # Reduce to num_bins using KMeans for representativeness
    if valid_ab_values.shape[0] > num_bins:
        kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(valid_ab_values)
        valid_ab_values = kmeans.cluster_centers_

    # Convert to a tensor
    quantized_bins = torch.tensor(valid_ab_values, dtype=torch.float32)

    # print(f'Quantized bins:\n{quantized_bins}')
    del kmeans, in_gamut_mask, rgb_colors,lab_colors, ab_grid, a_values, b_values
    return quantized_bins


# To obtain Z_ground_truth we define the relative mapping H^(-1)
def gaussian_weight(distances, sigma):
    return torch.exp(-distances ** 2 / (2 * sigma ** 2))


def inverse_h_mapping(ab_channels, quantized_colorspace_bins, sigma=5, chunk_size=200, k=5):
    """
            Mapping the ground truth (Y) onto a soft-encoded representation (probabilities for each bin)
            using the calculation of k-nearest neighbors (KNN) and a Gaussian distance.

            Parameters:
                ab_channels (Tensor): Tensor of the 'a' and 'b' components for each pixel in the image of size [Batch, 2, Height, Width].
                quantized_colorspace_bins (Tensor): Tensor containing the 313 quantized bins (Q = 313).
                sigma (float): Parameter for the calculation of the Gaussian weight (default 5).
                chunk_size (int): Number of pixels to process at a time to avoid memory issues (default 200).
                k (int): Number of nearest neighbors to consider (default 5).

            Returns:
                Tensor: A tensor representing the probability of each pixel belonging to each bin.
            """

    # Assume shape is [Batch, 2, Height, Width]
    B, C, H, W = ab_channels.shape
    assert C == 2, "Input ab_channels must have 2 channels (a, b)."

    '''
    print(f"###### INVERSE H MAPPING #####\nab_channels min: {ab_channels.min()}, max: {ab_channels.max()}")
    print(f"quantized_colorspace_bins min: {quantized_colorspace_bins.min()}, max: {quantized_colorspace_bins.max()}")
    print(f"Quantized colorspace bins shape: {quantized_colorspace_bins.shape}\n")
    print(f"Quantized colorspace bins sample: {quantized_colorspace_bins[:5]}\n")
    '''

    # Flatten spatial dimensions
    ab_flat = ab_channels.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)  # [N, 2]
    N = ab_flat.shape[0]  # Total number of pixels

    # Initialize the output tensor
    soft_encoded_flat = torch.zeros((N, quantized_colorspace_bins.shape[0]), device=ab_channels.device)  # [N, Q]
    z_scalar_flat = torch.zeros(N, dtype=torch.long, device=ab_channels.device)  # [N]

    # Process in chunks to avoid memory issues
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = ab_flat[start:end]  # Current chunk [chunk_size, 2]

        # Calculate distances between the chunk and all the quantized bins
        distances = torch.cdist(chunk, quantized_colorspace_bins, p=2)  # [chunk_size, Q]
        # Find the k-nearest neighbors (KNN) for each pixel
        knn_distances, knn_indices = torch.topk(distances, k=k, largest=False, dim=1)  # [chunk_size, k]

        # Debugging
        #print(f"Chunk {start}:{end} distances: {distances.min()}, {distances.max()}")
        #print(f"KNN distances: {knn_distances.min()}, {knn_distances.max()}\n")

        # Compute gaussian weights
        knn_weights = gaussian_weight(knn_distances, sigma)  # Gaussian kernel [chunk_size, k]
        # Normalize the weights to sum to 1
        normalized_knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True)  # [chunk_size, k]

        # Accumulate the weights into the soft-encoded bins
        for i in range(k):
            soft_encoded_flat[torch.arange(start, end), knn_indices[:, i]] += normalized_knn_weights[:, i]

        # This is not necessary, but it can be useful for debug:
        # Assign the nearest bin (scalar ground truth).
        z_scalar_flat[start:end] = knn_indices[:, 0]  # Take the closest bin for scalar encoding

    # Reshape the soft-encoded tensor back to [B, 313, H, W]
    soft_encoded = soft_encoded_flat.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, 313, H, W]
    z_scalar = z_scalar_flat.view(B, H, W)  # [B, H, W]

    '''
    print(f"Non-zero entries in soft_encoded: {(soft_encoded > 0).sum().item()}\n")
    print(f"Max value in soft_encoded: {soft_encoded.max()}\n\n")
    bin_sums = soft_encoded.sum(dim=(0, 2, 3))  # Sum over batch, height, and width
    print(f"Bin sums: {bin_sums}\n")
    print(f"Non-zero bins: {(bin_sums > 0).sum().item()}\n")
    print(f"Soft_encoded sum per pixel (should be ~1): {soft_encoded.sum(dim=1)}\n")

    # print(f'soft_encoded shape: {soft_encoded.shape}')
    print(f"Final soft_encoded sum: {soft_encoded.sum()}\n\n")
    print(f"Final soft_encoded: {soft_encoded}\n\n")
    '''
    del z_scalar_flat, soft_encoded_flat, distances, knn_distances, knn_indices, knn_weights
    return soft_encoded, z_scalar


def resize_to_64x64(image):
    """
    Resize the soft-encoded representation from size (224, 224, 313) to (64, 64, 313)
    using bilinear interpolation.

    soft_encoded_224x224: Tensor of size (B, 313, 224, 224), the soft-encoded representation.

    Returns a tensor of size (B, 313, 64, 64).
    """
    # Perform downsampling to 64x64
    image = image.permute(0, 2, 3, 1)  # (B, H, W, 313)
    image = image.permute(0, 3, 1, 2)  # (B, 313, H, W)

    # Downscale to 64x64
    output = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)

    return output

def point_estimate_H(AB_Predicted):
    T = 1  # Temperature adjustment (1 -> no changes)

    # Ensure input is strictly positive to avoid issues with log
    if torch.any(AB_Predicted <= 0):
        raise ValueError("Input tensor contains non-positive values, which are invalid for log.")

    # Compute normalized tensor directly
    num = torch.exp(torch.log(AB_Predicted) / T)  # Numerator
    den = torch.sum(num, dim=1, keepdim=True)  # Denominator: Sum over the channel dimension

    # Normalize num by den
    normalized = num / den

    # Compute mean over batch, width, and height dimensions
    return normalized
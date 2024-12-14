import torchvision
import torch
import torch.nn as nn
import numpy as np
from skimage import color
from PIL import Image
import torch.nn.functional as F
from skimage.color import lab2rgb
from sklearn.cluster import KMeans

class ZhangColorizationNetwork(nn.Module):
    def __init__(self):
        super(ZhangColorizationNetwork, self).__init__()

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
        image_pil = Image.fromarray(image)

        # Convert to RGB from RGBA if necessary
        if image.shape[-1] == 4:
            image_pil = image_pil.convert('RGB')

        return np.asarray(image_pil.resize((new_size[1], new_size[0]), resample=resample))

    @staticmethod
    def preprocess_img(img_rgb_orig, new_size=(256, 256), resample=3):

        img_rgb_rs = ZhangColorizationNetwork.res_image(img_rgb_orig, new_size=new_size, resample=resample)

        img_lab_orig = color.rgb2lab(img_rgb_orig)
        img_lab_rs = color.rgb2lab(img_rgb_rs)

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

    def point_estimate_H(self, Z_predicted):
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

    print("Started training...")
    # Compute the quantized bins and move them to the correct device
    quantized_colorspace = quantized_bins().to(device)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_loss_weighed = 0.0
        for l_resized, img_lab_orig in trainloader:
            # Get the LAB original image
            # print(f'Original lab image shape: {img_lab_orig.shape}')
            # print(f'L resized shape: {l_resized.shape}')

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

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)

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
        valid_losses.append(valid_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")


# Custom loss function
def multinomial_cross_entropy_loss_L(raw_network_output, z_ground_truth):
    # Apply softmax to ensure Z_predicted is a valid probability distribution
    z_predicted = torch.softmax(raw_network_output, dim=-1)

    # Masking to avoid getting a nan result from applying a log function to the z_predicted values
    mask = z_predicted > 0
    masked_log = torch.zeros_like(z_predicted, device=z_predicted.device)
    masked_log[mask] = torch.log(z_predicted[mask])

    # print(f'Z_ground_truth shape: {Z_ground_truth.shape}')
    # print(f'Z_predicted shape: {Z_predicted.shape}')

    # print()
    # print('Z_predicted:')
    # print(Z_predicted)
    # print()
    # print('Z_ground_truth')
    # print(Z_ground_truth)
    # print()

    loss = -torch.sum(z_ground_truth * masked_log)
    print(f'Loss: {loss}')

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

        # print(f'soft_encoded shape: {soft_encoded.shape}')

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


class LabNormalization:

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

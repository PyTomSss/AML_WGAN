import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

###### DISCRIMINATOR NETWORK 
# D(x) is the discriminator network which outputs the (scalar) probability that x
# came from training data rather than the generator. Here, sinum_channele we are dealing with 
# images, the input to D(x) is an image.

class Discriminator(nn.Module):
    def __init__(self, img_size, dim):
        """
        img_size : (int, int, int)
            img_size[0] : num_channels (e.g. 3 for RGB, 1 for grayscale)
            img_size[1] : Height (H)
            img_size[2] : Width (W)
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[0], dim, 4, 2, 1),  # First dimension is the number of channels
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        # Output size calculation after 4 convolutions with stride 2 (halving size each time)
        output_size = 8 * dim * (img_size[1] // 16) * (img_size[2] // 16)
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
 
    
###### GENERATOR NETWORK
# For the generator’s notation, let z be a latent space vector sampled from a standard normal 
# distribution. G(z) represents the generator funum_channeltion which maps the latent vector z to data-space. 
# The goal of G is to estimate the distribution that the training data comes from (p_data) so it 
# can generate fake samples from that estimated distribution (p_g).

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (self.img_size[1] // 16, self.img_size[2] // 16)  # Adjusted for (height, width)

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 8 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[0], 4, 2, 1),  # Adjusted for output channels
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # Map latent vector to appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape to (batch_size, 8 * dim, feature_height, feature_width)
        x = x.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        # Return the generated image
        return self.features_to_image(x)

    def init_weight(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))

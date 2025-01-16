import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import os

###### DISCRIMINATOR NETWORK 
# D(x) is the discriminator network which outputs the (scalar) probability that x
# came from training data rather than the generator. Here, sinum_channele we are dealing with 
# images, the input to D(x) is an image.

class WGANCPDiscriminator(nn.Module):
    def __init__(self, img_size, dim):
        """
        img_size : (int, int, int)
            img_size[0] : num_channels (e.g. 3 for RGB, 1 for grayscale)
            img_size[1] : Height (H)
            img_size[2] : Width (W)
        """
        super(WGANCPDiscriminator, self).__init__()

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

class WGANCPGenerator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(WGANCPGenerator, self).__init__()

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


class WGANCP_Trainer(object):
    def __init__(self, opt, dataloader):

        self.discriminator = WGANCPDiscriminator(opt["img_size"], opt["dim"])
        self.generator = WGANCPGenerator(opt["img_size"], opt["latent_dim"], opt["dim"])
        self.dataloader = dataloader

        # store accuracies and losses for plotting
        self.D_loss, self.G_loss, self.gradient_penalty_list = [], [], []
        self.epoch_times = []
        self.lr = opt["lr"] ## make it lower for wasserstein !!
        self.device = opt["device"]
        self.criterion = nn.BCELoss()
        self.batch_size = opt["batch_size"] 

        # WGAN with gradient clipping uses RMSprop instead of ADAM
        self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=self.lr)
        self.optimizer_G = optim.RMSprop(self.generator.parameters(), lr=self.lr)

        self.n_critic = opt["n_critic"]
        self.clip_val = opt["clip_val"]

    def train(self):

        self.discriminator.to(self.device)
        self.generator.to(self.device)
        # we want to compare the rapidity of the training
        total_time_start = t.time()

        for epoch in range(self.num_epochs):
            start_time = t.time()
            epoch_discriminator_loss = 0
            epoch_generator_loss = 0

            for i, (image, _) in enumerate(self.dataloader):

                real_imgs = image.to(self.device)

                #####################################################################################
                ################################ TRAIN DISCRIMINATOR ################################
                #####################################################################################

                for _ in range(self.n_critic):

                    z = torch.randn(real_imgs.size(0), self.generator.latent_dim).to(self.device)
                    fake_imgs = self.generator(z).detach()

                    d_loss = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(fake_imgs))

                    self.optimizer_D.zero_grad()
                    d_loss.backward()
                    self.optimizer_D.step()

                    # Clipping des poids du Critique
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-self.clip_val, self.clip_val)

                epoch_discriminator_loss += d_loss.item()

                #####################################################################################
                ################################## TRAIN GENERATOR ##################################
                #####################################################################################

                z = torch.randn(real_imgs.size(0), self.generator.latent_dim).to(self.device)
                fake_imgs = self.generator(z)
                g_loss = -torch.mean(self.discriminator(fake_imgs))

                epoch_generator_loss += g_loss.item()
                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()
                                # Print loss and accuracy every 50 iterations
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                        % (epoch + 1, self.num_epochs, i, len(self.dataloader), d_loss, g_loss))


            # save the time taken by the epoch
            self.epoch_times.append(t.time() - start_time)

            # save the losses values at the end of each epoch
            avg_discriminator_loss = epoch_discriminator_loss / len(self.dataloader)
            avg_generator_loss = epoch_generator_loss / len(self.dataloader)
            self.G_loss.append(avg_generator_loss)
            self.D_loss.append(avg_discriminator_loss)
        
        total_time_end = t.time()
        self.training_time = total_time_end - total_time_start
        print('Time of training-{}'.format((self.training_time)))

        save_path_generator=f"../trained_models/WGANCPgenerator_epoch{self.num_epochs}.pth"
        save_path_discriminator=f"../trained_models/WGANCPdiscriminator_epoch{self.num_epochs}.pth"
        # Save the trained models
        torch.save(self.generator.state_dict(), save_path_generator)
        torch.save(self.discriminator.state_dict(), save_path_discriminator)
        print(f"Models saved: Generator -> {save_path_generator}, Discriminator -> {save_path_discriminator}")

        # Plot the generator and discriminator losses
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_loss, label="G")
        plt.plot(self.D_loss, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        return self.G_loss, self.D_loss, self.epoch_times, self.training_time
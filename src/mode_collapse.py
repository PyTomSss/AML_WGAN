import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


class GANLoss_G(nn.Module):
    """
    This class implements the standard generator GAN loss proposed in:
    https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf
    """

    def __init__(self):
        """
        Constructor method.
        """
        # Call super constructor
        super(GANLoss_G, self).__init__()

    def forward(self, D_fake_pred, **kwargs):
        return - F.softplus(D_fake_pred).mean()


class GANLoss_D(nn.Module):

    def __init__(self):
        super(GANLoss_D, self).__init__()

    def forward(self, D_real_pred, D_fake_pred, **kwargs):
        return F.softplus(- D_real_pred).mean() + \
            F.softplus(D_fake_pred).mean()


class WGANLoss_G(nn.Module):
    """
    This class implements the Wasserstein generator GAN loss proposed in:
    http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf
    """

    def __index__(self):
        super(WGANLoss_G, self).__index__()

    def forward(self, D_fake_pred, **kwargs):

        return - D_fake_pred.mean()


class WGANLoss_D(nn.Module):

    def __init__(self):
        super(WGANLoss_D, self).__init__()

    def forward(self, D_real_pred, D_fake_pred, **kwargs):

        return - D_real_pred.mean() + D_fake_pred.mean()


class WGANGPLoss_G(WGANLoss_G):
    """
    This class implements the Wasserstein generator GAN loss proposed in:
    https://proceedings.neurips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf
    """

    def __index__(self):
        super(WGANGPLoss_G, self).__index__()


class WGANGPLoss_D(nn.Module):
    """
    This class implements the Wasserstein generator GAN loss proposed in:
    https://proceedings.neurips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf
    """

    def __init__(self):
        """
        Constructor method.
        """
        # Call super constructor
        super(WGANGPLoss_D, self).__init__()

    def forward(self, D_real_red, D_fake_pred, D, real_samples, fake_samples, lambda_gp = 2., **kwargs):
        """
        Forward pass.
        :param discriminator_prediction_real: (torch.Tensor) Raw discriminator prediction for real samples
        :param discriminator_prediction_fake: (torch.Tensor) Raw discriminator predictions for fake samples
        :return: (torch.Tensor) Wasserstein discriminator GAN loss with gradient penalty
        """
        # Generate random alpha for interpolation
        alpha = torch.rand((real_samples.shape[0], 1), device=real_samples.device)
        # Make interpolated samples
        samples_interpolated = (alpha * real_samples + (1. - alpha) * fake_samples)
        samples_interpolated.requires_grad = True
        # Make discriminator prediction
        discriminator_prediction_interpolated = D(samples_interpolated)
        # Calc gradients
        gradients = torch.autograd.grad(outputs=discriminator_prediction_interpolated.sum(),
                                        inputs=samples_interpolated,
                                        create_graph=True,
                                        retain_graph=True)[0]
        # Calc gradient penalty
        gradient_penalty = (gradients.view(gradients.shape[0], -1).norm(dim=1) - 1.).pow(2).mean()
        return - D_real_red.mean() \
               + D_fake_pred.mean() \
               + lambda_gp * gradient_penalty



class mode_collapse_experiment(object):

    def __init__(self, opt):
        self.latent_size = opt["latent_size"]
        self.device = opt["device"]
        self.lr = opt["lr"]
        self.num_epochs = opt["num_epochs"]
        self.batch_size = opt["batch_size"]
        self.n_critic = opt["n_critic"]
        self.clip_w = opt["clip_w"]

        # params used for gaussian ring simulation
        self.num_samples = opt["num_samples"] # fixed in this experience
        self.gr_variance = 0.05 # fixed in this experience
        self.data = self.generate_gaussian_ring().to(self.device)
        self.generator = self.G().to(self.device)
        self.discriminator = self.D().to(self.device)

        self.mode = opt['mode'] # options : "normal" (GAN), "wasserstein" (clipping), "wasserstein-gp" (gradient penalty)

        self.optim_G = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.optim_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

        self.img_plot_periodicity = opt["img_plot_periodicity"]

    def G(self):
        """
        Returns the generator network.
        :param latent_size: (int) Size of the latent input vector
        :return: (nn.Module) Simple feed forward neural network with three layers,
        """
        return nn.Sequential(nn.Linear(self.latent_size, 256, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(256, 256, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(256, 256, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(256, 256, bias=True),
                            nn.Tanh(),
                            nn.Linear(256, 2, bias=True))


    def D(self):
        """
        Returns the discriminator network.
        return: Simple feed forward neural network with three layers and probability output.
        """
        return nn.Sequential(nn.Linear(2, 256, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(256, 256, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(256, 256, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(256, 256, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(256, 1, bias=True))


    def generate_gaussian_ring(self):
        angles = torch.cumsum((2 * np.pi / 8) * torch.ones((8)), dim=0)
        # Convert angles to 2D coordinates
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=0)
        # Generate data
        data = torch.empty((2, self.num_samples))
        counter = 0
        for gaussian in range(means.shape[1]):
            for sample in range(int(self.num_samples / 8)):
                data[:, counter] = torch.normal(means[:, gaussian], self.gr_variance)
                counter += 1
        # Reshape data
        data = data.T
        # Shuffle data
        data = data[torch.randperm(data.shape[0])]
        # Convert numpy array to tensor
        return data.float()


    def mode_collapse(self):
        # Make directory to save plots
        # path = os.path.join(os.getcwd(), 'plots', args.loss + ("_top_k" if args.topk else "") + ("_sn" if args.spectral_norm else "") + ("_clip" if args.clip_weights else ""))
        # os.makedirs(path, exist_ok=True)
        # Init hyperparameters
        fixed_generator_noise = torch.randn([self.num_samples // 10, self.latent_size], device=self.device)
        # Get data
        # Init Loss function
        if self.mode == 'normal':
            loss_generator = GANLoss_G()
            loss_discriminator = GANLoss_D()
        elif self.mode == 'wasserstein':
            loss_generator = WGANLoss_G()
            loss_discriminator = WGANLoss_D()
        elif self.mode == 'wasserstein-gp':
            loss_generator = WGANGPLoss_G()
            loss_discriminator = WGANGPLoss_D()

        # Networks to train mode
        self.generator.train()
        self.discriminator.train()

        # Models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        data = self.data

        # Training loop
        for epoch in tqdm(range(self.num_epochs), desc="Epoch Progress", ncols=100):

            for index in range(0, self.num_samples, self.batch_size):  # type:int
                # Shuffle data
                data = data[torch.randperm(data.shape[0], device=self.device)]
                
                #####################################################################################
                ################################ TRAIN DISCRIMINATOR ################################
                #####################################################################################

                # Update discriminator more often than generator to train it till optimality and get more reliable gradients of Wasserstein
                for _ in range(self.n_critic):  
                    # Get batch
                    batch = data[index:index + self.batch_size]
                    # Get noise for generator
                    noise = torch.randn([self.batch_size, self.latent_size], device=self.device)
                    # Optimize discriminator
                    self.optim_D.zero_grad()
                    self.optim_G.zero_grad()
                    
                    with torch.no_grad():
                        fake_samples = self.generator(noise)

                    real_pred = self.discriminator(batch)
                    fake_pred = self.discriminator(fake_samples)

                    if self.mode == "wasserstein-gp":
                        loss_d = loss_discriminator(real_pred, fake_pred, self.discriminator, batch,
                                                                fake_samples)
                    else:
                        loss_d = loss_discriminator(real_pred, fake_pred)
                    loss_d.backward()
                    self.optim_D.step()

                    # Clip weights to enforce Lipschitz constraint as proposed in Wasserstein GAN paper
                    if self.clip_w > 0:
                        with torch.no_grad():
                            for param in self.discriminator.parameters():
                                param.clamp_(-self.clip_w, self.clip_w)

                #####################################################################################
                ################################## TRAIN GENERATOR ##################################
                #####################################################################################

                noise = torch.randn([self.batch_size, self.latent_size], device=self.device)

                self.optim_D.zero_grad()
                self.optim_G.zero_grad()

                fake_samples = self.generator(noise)
                fake_pred = self.discriminator(fake_samples)
                
                loss_g = loss_generator(fake_pred)
                loss_g.backward()
                self.optim_G.step()
                
            # Plot samples of generator
            if ((epoch + 1) % self.img_plot_periodicity) == 0 or epoch == 0:
                self.generator.eval()
                generator_samples = self.generator(fixed_generator_noise)
                generator_samples = generator_samples.cpu().detach().numpy()
                fig, ax = plt.subplots(figsize=(15, 7))
                
                # Plot real data samples
                ax.scatter(
                    data[::10, 0].cpu(),  # Subsample data for visibility
                    data[::10, 1].cpu(),
                    color='blue', 
                    label=r'Samples from $p_{data}$', 
                    s=2, 
                    alpha=0.5
                )
                
                # Plot generated samples
                ax.scatter(
                    generator_samples[:, 0], 
                    generator_samples[:, 1], 
                    color='red', 
                    label=r'Samples from generator $G$', 
                    s=2, 
                    alpha=0.5
                )
                
                # Add legend
                ax.legend(loc='upper right', fontsize=12)
                
                # Add title with epoch and step information
                ax.set_title(f"Mode Collapse For {self.mode} (step: {(epoch + 1) * self.num_samples // self.batch_size})", fontsize=16)
                
                # Set axis limits
                ax.set_xlim((-1.5, 1.5))
                ax.set_ylim((-1.5, 1.75))
                
                # Add grid for better visualization
                ax.grid(alpha=0.3)
                
                # Save the plot
                filename = f'mode_collapse_type:{self.mode}_epoch:{self.num_epochs}'
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
                self.generator.train()
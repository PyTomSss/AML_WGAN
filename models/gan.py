import os
import time
import torch
import torch.nn as nn
from torchvision import utils
from torch.autograd import Variable
from utils.tensorboard_logger import Logger

class GAN(object):
    def __init__(self, opt):
        # Define the Generator neural network architecture
        self.generator = nn.Sequential(
            nn.Linear(100, 256),          # Input layer with 100 features, output 256
            nn.LeakyReLU(0.2),            # Activation function with negative slope
            nn.Linear(256, 512),          # Hidden layer: 256 -> 512 features
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),         # Hidden layer: 512 -> 1024 features
            nn.LeakyReLU(0.2),
            nn.Tanh()                     # Output activation for Generator
        )

        # Define the Discriminator neural network architecture
        self.discriminator = nn.Sequential(
            nn.Linear(1024, 512),         # Input layer with 1024 features, output 512
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),          # Hidden layer: 512 -> 256 features
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),            # Output layer: 256 -> 1 (probability of real or fake)
            nn.Sigmoid()                  # Sigmoid activation to output probability
        )

        # CUDA configuration for GPU support
        self.cuda = opt["cuda"]
        self.device = opt["device"]

        # Binary Cross-Entropy Loss for both Generator and Discriminator
        self.loss = nn.BCELoss().to(self.device)
        self.G_loss, self.D_loss = []

        # Optimizers for the Discriminator and Generator with learning rate and weight decay
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, weight_decay=0.00001)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, weight_decay=0.00001)

        # Logger setup for tracking training progress
        self.logger = Logger('./logs')
        self.number_of_images = 10       # Number of images to log
        self.epochs = opt["num_epochs"]# Total training epochs
        self.batch_size = opt["batch_size"]  # Training batch size

        self.discriminator.to(self.device)     # Move Discriminator to GPU
        self.generator.to(self.device)

    # Train the GAN
    def train(self, train_loader):
        self.t_begin = time.time()           # Record training start time
        generator_iter = 0                   # Counter for Generator iterations

        for epoch in range(self.epochs + 1): # Iterate through epochs
            for i, (images, _) in enumerate(train_loader):
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break  # Stop if the end of the batch is reached
                
                # Preprocess the input images (flattening them to 1D with 1024 features)
                images = images.view(self.batch_size, -1)
                z = torch.rand((self.batch_size, 100))  # Random noise for Generator input

                # Move tensors to GPU if CUDA is enabled
                if self.cuda :
                    real_labels = Variable(torch.ones(self.batch_size)).to(self.device)  # Real labels
                    fake_labels = Variable(torch.zeros(self.batch_size)).to(self.device)  # Fake labels
                    images, z = Variable(images.to(self.device)), Variable(z.to(self.device))
                else:
                    real_labels = Variable(torch.ones(self.batch_size))  # Real labels
                    fake_labels = Variable(torch.zeros(self.batch_size))  # Fake labels
                    images, z = Variable(images), Variable(z)

                # --- Train the Discriminator ---
                outputs = self.discriminator(images)                            # Discriminator output on real images
                d_loss_real = self.loss(outputs.flatten(), real_labels)  # Loss for real images
                real_score = outputs                                # Save real image scores

                fake_images = self.generator(z)                             # Generate fake images using Generator
                outputs = self.discriminator(fake_images)                      # Discriminator output on fake images
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)  # Loss for fake images
                fake_score = outputs                                # Save fake image scores

                d_loss = d_loss_real + d_loss_fake                 # Total Discriminator loss
                self.discriminator.zero_grad()                                 # Zero the gradients
                d_loss.backward()                                  # Backpropagation
                self.discriminator_optimizer.step()                            # Update Discriminator weights

                # --- Train the Generator ---
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100).to(self.device))  # Random noise
                else:
                    z = Variable(torch.randn(self.batch_size, 100))
                fake_images = self.generator(z)                             # Generate fake images
                outputs = self.discriminator(fake_images)                      # Discriminator output on fake images
                g_loss = self.loss(outputs.flatten(), real_labels)  # Generator loss (goal: fool Discriminator)

                self.discriminator.zero_grad()                                 # Zero the gradients
                self.generator.zero_grad()
                g_loss.backward()                                  # Backpropagation
                self.generator_optimizer.step()                            # Update Generator weights
                generator_iter += 1                                # Increment Generator iteration

                self.G_loss.append(g_loss.item())
                self.D_loss.append(d_loss.item())
                # Log training progress every 100 iterations
                if ((i + 1) % 100) == 0:
                    print(f"Epoch: [{epoch+1}] [{i+1}/{train_loader.dataset.__len__() // self.batch_size}] "
                          f"D_loss: {d_loss.data:.8f}, G_loss: {g_loss.data:.8f}")
                    
                    # Log metrics to TensorBoard
                    info = {
                        'd_loss': d_loss.data,
                        'g_loss': g_loss.data
                    }
                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, i + 1)

                # Save models periodically
                if generator_iter % 1000 == 0:
                    print('Generator iter-', generator_iter)
                    self.save_model()

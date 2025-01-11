import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, discriminator, generator, dataloader, lr, beta1, beta2, mode="normal", lambda_gp=10):
        """
        Args:
            discriminator: The discriminator model.
            generator: The generator model.
            dataloader: The dataloader for training data.
            lr: Learning rate.
            beta1: Beta1 parameter for Adam optimizer.
            beta2: Beta2 parameter for Adam optimizer.
            mode: "normal" or "wasserstein" to choose the training type.
            lambda_gp: Coefficient for gradient penalty in WGAN-GP.
        """
        self.discriminator = discriminator
        self.generator = generator
        self.dataloader = dataloader
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.mode = mode
        self.lambda_gp = lambda_gp

        # Optimizers for discriminator and generator
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        # Loss function for normal GAN
        if self.mode == "normal":
            self.criterion = nn.BCELoss()

    def gradient_penalty(self, real_data, fake_data):
        """
        Computes the gradient penalty for WGAN-GP.

        Args:
            real_data: Batch of real data samples.
            fake_data: Batch of generated data samples.

        Returns:
            Gradient penalty term.
        """
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real_data.device)  # Random weight for interpolation
        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates.requires_grad_(True)

        # Compute discriminator output for interpolates
        d_interpolates = self.discriminator(interpolates)
        gradients = grad(outputs=d_interpolates,
                         inputs=interpolates,
                         grad_outputs=torch.ones_like(d_interpolates),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        # Reshape and compute gradient norm
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_discriminator(self, real_data, fake_data):
        """
        Trains the discriminator on real and fake data.

        Args:
            real_data: Batch of real data samples.
            fake_data: Batch of generated data samples.

        Returns:
            The loss value for the discriminator.
        """
        self.optimizer_D.zero_grad()

        if self.mode == "normal":
            # Labels for real and fake data
            real_labels = torch.ones(real_data.size(0), 1, device=real_data.device)
            fake_labels = torch.zeros(fake_data.size(0), 1, device=fake_data.device)

            # Loss for real and fake data
            real_loss = self.criterion(self.discriminator(real_data), real_labels)
            fake_loss = self.criterion(self.discriminator(fake_data.detach()), fake_labels)

            d_loss = real_loss + fake_loss

        elif self.mode == "wasserstein":
            # Wasserstein loss (critic score difference)
            d_loss = -(self.discriminator(real_data).mean() - self.discriminator(fake_data.detach()).mean())

            # Add gradient penalty
            gp = self.gradient_penalty(real_data, fake_data)
            d_loss += self.lambda_gp * gp

        # Backpropagation and optimizer step
        d_loss.backward()
        self.optimizer_D.step()
        return d_loss.item()

    def train_generator(self, fake_data):
        """
        Trains the generator to produce realistic data.

        Args:
            fake_data: Batch of generated data samples.

        Returns:
            The loss value for the generator.
        """
        self.optimizer_G.zero_grad()

        if self.mode == "normal":
            # Labels for fake data (treated as real)
            real_labels = torch.ones(fake_data.size(0), 1, device=fake_data.device)
            g_loss = self.criterion(self.discriminator(fake_data), real_labels)

        elif self.mode == "wasserstein":
            # Wasserstein loss (maximize critic score)
            g_loss = -self.discriminator(fake_data).mean()

        # Backpropagation and optimizer step
        g_loss.backward()
        self.optimizer_G.step()
        return g_loss.item()

    def train(self, num_epochs, device):
        """
        Main training loop for the GAN, including discriminator accuracy calculation.

        Args:
            num_epochs: Number of training epochs.
            device: Device to run the training on (e.g., "cuda" or "cpu").
            save_path_generator: Path to save the trained generator model.
            save_path_discriminator: Path to save the trained discriminator model.
        """
        self.discriminator.to(device)
        self.generator.to(device)

        # Store losses for plotting
        G_losses = []
        D_losses = []
        D_accuracies = []  # Store discriminator accuracies for plotting

        for epoch in range(num_epochs):
            # Initialize variables for calculating accuracy
            correct_real = 0
            correct_fake = 0
            total = 0

            for i, (real_data, _) in enumerate(self.dataloader):
                real_data = real_data.to(device)
                batch_size = real_data.size(0)

                # Generate fake data
                noise = self.generator.init_weight(batch_size).to(device)
                fake_data = self.generator(noise)

                # Train discriminator
                d_loss = self.train_discriminator(real_data, fake_data)

                # Train generator every n_critic steps (1 step for normal GANs)
                if self.mode == "wasserstein":
                    n_critic = 5
                else:
                    n_critic = 1

                if i % n_critic == 0:
                    g_loss = self.train_generator(fake_data)

                # Save losses for plotting
                G_losses.append(g_loss)
                D_losses.append(d_loss)

                # Calculate accuracy for the discriminator
                real_preds = self.discriminator(real_data)
                fake_preds = self.discriminator(fake_data.detach())  # Detach fake data to avoid gradients affecting the generator

                # Count correct real and fake predictions
                correct_real += (real_preds > 0.5).sum().item()  # Real images should be classified as 1
                correct_fake += (fake_preds < 0.5).sum().item()  # Fake images should be classified as 0
                total += batch_size

                # Print loss and accuracy every 50 iterations
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tAccuracy_D: %.4f'
                        % (epoch + 1, num_epochs, i, len(self.dataloader), d_loss, g_loss, 100 * (correct_real + correct_fake) / total))

            # Calculate overall discriminator accuracy for the epoch
            epoch_accuracy = 100 * (correct_real + correct_fake) / total
            D_accuracies.append(epoch_accuracy)

        # Print accuracy after each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Discriminator Accuracy: {epoch_accuracy:.2f}%")

        save_path_generator=f"../models/generator_epoch{num_epochs}.pth"
        save_path_discriminator=f"../models/discriminator_epoch{num_epochs}.pth"
        # Save the trained models
        torch.save(self.generator.state_dict(), save_path_generator)
        torch.save(self.discriminator.state_dict(), save_path_discriminator)
        print(f"Models saved: Generator -> {save_path_generator}, Discriminator -> {save_path_discriminator}")

        # Plot the generator and discriminator losses
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        return G_losses, D_losses, D_accuracies

# Exemple d'utilisation :
# trainer = Trainer(discriminator, generator, dataloader, lr=0.0002, beta1=0.5, beta2=0.999, mode="wasserstein")
# trainer.train(num_epochs=100, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

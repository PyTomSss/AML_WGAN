import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
from torch.autograd import Variable

class Trainer:
    def __init__(self, discriminator, generator, dataloader, lr, beta1, beta2, device, mode="normal", lambda_gp=10, gp_weight=10):
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
        # store accuracies and losses for plotting
        self.D_loss, self.G_loss, self.gradient_penalty_list, self.D_accuracies = [], [], [], []
        self.lr = lr
        self.gp_weight = gp_weight
        self.beta1 = beta1
        self.beta2 = beta2
        self.mode = mode
        self.lambda_gp = lambda_gp
        self.device = device
        self.criterion = nn.BCELoss()

        # Optimizers for discriminator and generator
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        # Loss function for normal GAN
        if self.mode == "normal":
            self.criterion = nn.BCELoss()

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.generator.latent_dim))

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.sample_latent(num_samples)).to(self.device)
        generated_data = self.generator(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
   
    def gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(self.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def train_discriminator(self, real_data, generated_data):
        """
        Trains the discriminator on real and fake data.

        Args:
            real_data: Batch of real data samples.
            fake_data: Batch of generated data samples.

        Returns:
            The loss value for the discriminator.
        """
        self.optimizer_D.zero_grad()
        real_data = Variable(real_data)
        d_real = self.discriminator(real_data)
        d_generated = self.discriminator(generated_data)

        if self.mode == "normal":
            # Labels for real and fake data
            real_labels = torch.ones(real_data.size(0), 1, device=self.device)
            fake_labels = torch.zeros(generated_data.size(0), 1, device=self.device)

            # Loss for real and fake data
            real_loss = self.criterion(d_real, real_labels)
            fake_loss = self.criterion(d_generated, fake_labels)

            # self.discriminator(fake_data.detach())
            d_loss = real_loss + fake_loss

            # Backpropagation and optimizer step
            d_loss.backward()
            self.optimizer_D.step()
            return d_loss.item()
        
        elif self.mode == "wasserstein":
            # compute gradient penalty
            gp = self.gradient_penalty(real_data, generated_data)
            self.gradient_penalty_list.append(gp.item())

            self.optimizer_D.zero_grad()
            d_loss = d_generated.mean() - d_real.mean() + gp
        
            # Backpropagation and optimizer step
            d_loss.backward()
            self.optimizer_D.step()
            return d_loss.item()


    def train_generator(self, generated_data):
        """
        Trains the generator to produce realistic data.

        Args:
            fake_data: Batch of generated data samples.

        Returns:
            The loss value for the generator.
        """
        self.optimizer_G.zero_grad()
        d_generated = self.discriminator(generated_data) 

        if self.mode == "normal":
            # Labels for fake data (treated as real)
            real_labels = torch.ones(generated_data.size(0), 1, device=self.device)
            g_loss = self.criterion(self.discriminator(generated_data), real_labels)
            return g_loss.item()


        elif self.mode == "wasserstein":
            # Wasserstein loss (maximize critic score)
            d_generated = self.discriminator(generated_data)
            g_loss = - d_generated.mean()

            # Backpropagation and optimizer step
            g_loss.backward(retain_graph=True)
            self.optimizer_G.step()
            return g_loss.item()

    def train(self, num_epochs):
        """
        Main training loop for the GAN, including discriminator accuracy calculation.

        Args:
            num_epochs: Number of training epochs.
            device: Device to run the training on (e.g., "cuda" or "cpu").
            save_path_generator: Path to save the trained generator model.
            save_path_discriminator: Path to save the trained discriminator model.
        """
        self.discriminator.to(self.device)
        self.generator.to(self.device)

        for epoch in range(num_epochs):
            # Initialize variables for calculating accuracy
            correct_real = 0
            correct_fake = 0
            total = 0

            for i, (real_data, _) in enumerate(self.dataloader):
                real_data = real_data.to(self.device)
                # Generate fake data
                batch_size = real_data.size(0)
                generated_data = self.sample_generator(batch_size).to(self.device)

                # Train discriminator
                d_loss = self.train_discriminator(real_data=real_data, generated_data=generated_data)

                # Train generator every n_critic steps (1 step for normal GANs)
                if self.mode == "wasserstein":
                    n_critic = 5
                else:
                    n_critic = 1

                if i % n_critic == 0:
                    g_loss = self.train_generator(generated_data=generated_data)

                # Save losses for plotting
                self.G_loss.append(g_loss)
                self.D_loss.append(d_loss)

                # Calculate accuracy for the discriminator
                real_preds = self.discriminator(real_data)
                fake_preds = self.discriminator(generated_data.detach())  # Detach fake data to avoid gradients affecting the generator

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
            self.D_accuracies.append(epoch_accuracy)

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
        plt.plot(self.G_loss, label="G")
        plt.plot(self.D_loss, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        return self.G_loss, self.D_loss, self.D_accuracies

# Exemple d'utilisation :
# trainer = Trainer(discriminator, generator, dataloader, lr=0.0002, beta1=0.5, beta2=0.999, mode="wasserstein")
# trainer.train(num_epochs=100, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

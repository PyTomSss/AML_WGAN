from src.train import Trainer
from src.gan_models import Generator, Discriminator
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def main(opt):

    # Définir les transformations : redimensionnement et conversion en tenseur
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convertir en tenseur
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation
    ])

    # Charger le dataset CIFAR-10 (train)
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Créer un DataLoader pour charger les données
    train_loader = DataLoader(train_dataset, batch_size=opt["batch_size"], shuffle=True)

    discriminator = Discriminator(opt["img_size"], opt["dim"])
    generator = Generator(opt["img_size"], opt["latent_dim"], opt["dim"])

    trainer = Trainer(discriminator, generator, train_loader, opt["lr"], opt["beta1"], opt["beta2"], opt["device"], mode="normal", lambda_gp=10)

    G_losses, D_losses, D_accuracies = trainer.train(opt["num_epochs"], opt["device"])

    return G_losses, D_losses, D_accuracies
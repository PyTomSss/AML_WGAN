from models.dcgan import DCGAN_Trainer
from models.wgan_cp import WGANCP_Trainer

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
    train_dataloader = DataLoader(train_dataset, batch_size=opt["batch_size"], shuffle=True)

    if opt["mode"] == "normal":
        trainer = DCGAN_Trainer(opt, train_dataloader)
    elif opt["mode"] == "wasserstein":
        trainer = WGANCP_Trainer(opt, train_dataloader)
    
    else: 
        print("mode not recognized")


    G_losses, D_losses, epoch_times, training_times = trainer.train()

    return G_losses, D_losses, epoch_times, training_times
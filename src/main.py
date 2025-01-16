from models.dcgan import DCGAN_Trainer
from models.wgan_cp import WGANCP_Trainer
from models.wgan_gp import WGANGP_Trainer

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import numpy as np
import os

def main(opt):

    # Charger le dataset CIFAR-10 (train)
    if opt["dataset_type"] == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        # Define the class index for "horse"
        horse_class_index = 7
        cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        # Get the indices of all "horse" class images
        horse_indices = [i for i, (_, label) in enumerate(cifar10_dataset) if label == horse_class_index]

        # Create a subset dataset for the "horse" class
        train_dataset = Subset(cifar10_dataset, horse_indices)

    elif opt["dataset_type"] == "celebA":
        transform = transforms.Compose([
                                        transforms.Resize((128, 128)),  # Redimensionner les images à une taille uniforme (par exemple 128x128)
                                        transforms.ToTensor(),  # Convertir les images en tenseurs
                                        ])

        opt["img_size"] = (3, 128, 128)

        # Chargement du dataset CelebA
        celeba_dataset = datasets.CelebA(
            root='./data',
            split='train',
            download=True,
            transform=transform)

        # Charge le fichier identity pour obtenir les labels de célébrités
        identity_file = os.path.join(celeba_dataset.root, 'celeba', 'identity_CelebA.txt')

        # ID de Tom Cruise dans le dataset CelebA
        target_celebrity_id = 1  # Tom Cruise a l'ID 1 dans CelebA

        # Lire les identités des célébrités
        celebrity_ids = np.loadtxt(identity_file, dtype=int)

        # Trouver les indices des images de Tom Cruise
        celebrity_indices = np.where(celebrity_ids == target_celebrity_id)[0]

        # Créer un sous-ensemble du dataset pour ne garder que les images de Tom Cruise
        train_dataset = Subset(celeba_dataset, celebrity_indices)

    # Créer un DataLoader pour charger les données
    train_dataloader = DataLoader(train_dataset, batch_size=opt["batch_size"], shuffle=True)

    if opt["mode"] == "normal":
        trainer = DCGAN_Trainer(opt, train_dataloader)
    elif opt["mode"] == "wasserstein-cp":
        trainer = WGANCP_Trainer(opt, train_dataloader)
    elif opt["mode"] == "wasserstein-gp":
        trainer = WGANGP_Trainer(opt, train_dataloader)
    else:
        print("mode not recognized")

    G_losses, D_losses, epoch_times, training_times = trainer.train()

    return G_losses, D_losses, epoch_times, training_times
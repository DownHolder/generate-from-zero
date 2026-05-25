import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, channels, image_size):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        input_dim = latent_dim + num_classes
        image_dim = channels * image_size * image_size

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_dim),
            nn.Tanh(),
        )

        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size

    def forward(self, z, labels):
        label_embed = self.label_embedding(labels)
        x = torch.cat([z, label_embed], dim=1)
        images = self.net(x)
        return images.view(-1, self.channels, self.image_size, self.image_size)


class Discriminator(nn.Module):
    def __init__(self, num_classes, channels, image_size):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        image_dim = channels * image_size * image_size
        input_dim = image_dim + num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, images, labels):
        label_embed = self.label_embedding(labels)
        x = torch.cat([images.view(images.size(0), -1), label_embed], dim=1)
        return self.net(x)

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, channels, image_size):
        super().__init__()
        image_dim = channels * image_size * image_size

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
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

    def forward(self, z):
        images = self.net(z)
        return images.view(-1, self.channels, self.image_size, self.image_size)


class Discriminator(nn.Module):
    def __init__(self, channels, image_size):
        super().__init__()
        image_dim = channels * image_size * image_size

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, images):
        return self.net(images)

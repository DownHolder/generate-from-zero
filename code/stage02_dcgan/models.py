import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, channels, image_size, features=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        self.features = features

        mult = features * 4
        self.init_size = image_size // 4  # 28 -> 7

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, mult * self.init_size * self.init_size),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(mult),
            nn.ConvTranspose2d(mult, features * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features * 2),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(features),
            nn.ConvTranspose2d(features, channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), self.features * 4, self.init_size, self.init_size)
        return self.upsample(out)


class Discriminator(nn.Module):
    def __init__(self, channels, image_size, features=64):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.features = features

        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, 1, 3, 1, 0),
            nn.Flatten(),
        )

    def forward(self, images):
        return self.net(images)

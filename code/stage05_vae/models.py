import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, image_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.logvar_head(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, image_dim, channels, image_size):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, image_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, self.channels, self.image_size, self.image_size)


class VAE(nn.Module):
    def __init__(self, latent_dim, channels, image_size):
        super().__init__()
        image_dim = channels * image_size * image_size
        self.encoder = Encoder(image_dim, latent_dim)
        self.decoder = Decoder(latent_dim, image_dim, channels, image_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

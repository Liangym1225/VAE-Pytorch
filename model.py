import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = encoder(latent_dim=latent_dim)
        self.decoder = decoder(latent_dim=latent_dim)

    def forward(self, x):
        mu, logsig = self.encoder(x)
        # reparameterization trick
        err = torch.rand_like(logsig)
        z = mu + torch.exp(logsig / 2) * err
        y = self.decoder(z)

        return z, mu, logsig, y


class encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(encoder, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=28 * 28, out_features=512)
        self.act1 = nn.ReLU(inplace=True)
        self.linear_mu = nn.Linear(in_features=512, out_features=latent_dim)
        self.linear_logsig = nn.Linear(in_features=512, out_features=latent_dim)

    def forward(self, x):
        x = self.act1(self.linear1(self.flatten(x)))
        mu = self.linear_mu(x)
        logsig = self.linear_logsig(x)  # log sigma^2

        return mu, logsig


class decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(decoder, self).__init__()
        self.linear1 = nn.Linear(in_features=latent_dim, out_features=512)
        self.act1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_features=512, out_features=28 * 28)

    def forward(self, z):
        z = self.act1(self.linear1(z))
        y = torch.sigmoid(self.linear2(z))
        y = y.view(-1, 1, 28, 28)
        return y

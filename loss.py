import torch
import torch.nn as nn


class ELBO(nn.Module):
    def __init__(self, distribution="gaussian", latent_dim=2):
        super(ELBO, self).__init__()
        self.distribution = distribution
        self.latent_dim = latent_dim
        self.reconstuctionloss = Reconstuctionloss(distribution=self.distribution)
        self.kl_divergence= KL_divergence(latent_dim=self.latent_dim)

    def forward(self, mu, logsig, output, gt):
        rcloss = self.reconstuctionloss(output,gt)
        kldiv = self.kl_divergence(mu, logsig)
        return rcloss+kldiv, rcloss.item(), kldiv.item()


class Reconstuctionloss(nn.Module):
    def __init__(self, distribution="mse"):
        super(Reconstuctionloss, self).__init__()
        if distribution == "gaussian":
            self.loss = nn.MSELoss(reduction="sum")
        elif distribution == "binomial":
            self.loss = nn.BCELoss(reduction="sum")

    def forward(self, output, gt):
        return self.loss(output, gt)


class KL_divergence(nn.Module):
    def __init__(self, latent_dim=2):
        super(KL_divergence, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, mu, logsig):
        return  - 0.5 *torch.sum(1 + logsig - mu.pow(2) - logsig.exp())

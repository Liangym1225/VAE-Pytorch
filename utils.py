import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def random_sampling(model, latent_dim, img_size, num, device):
    z = torch.randn(num**2, latent_dim).to(device)
    model.eval()
    with torch.no_grad():
        output = model.decoder(z)

    output=output.cpu().detach().numpy()
    output = np.squeeze(output)
    img = np.zeros((num * img_size, num * img_size))
    for y in range(num):
        for x in range(num):
            idx = y * num + x
            img[
                y * img_size : (y + 1) * img_size, x * img_size : (x + 1) * img_size
            ] = output[idx]*255
    img = Image.fromarray(img).convert("L")
    return img


def plot_manifold(model, z0, z1, img_size, num, device):
    img = np.zeros((num * img_size, num * img_size))
    model.eval()
    with torch.no_grad():
        for i, y in enumerate(np.linspace(z1[0], z1[1], num)):
            for j, x in enumerate(np.linspace(z0[0], z0[1], num)):
                z = torch.Tensor(([[x, y]])).to(device)
                output = model.decoder(z)
                output = output.cpu().detach().numpy()
                img[
                    (num - 1 - i) * img_size : (num - i) * img_size,
                    j * img_size : (j + 1) * img_size,
                ] = (
                    output * 255
                )

    img = Image.fromarray(img).convert("L")
    return img


def plot_latent_space(model, epoch, dataloader, num_batches, root):
    plt.figure()
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):

            z, _, _, _ = model(X)
            z = z.cpu().detach().numpy()
            y = y.tolist()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")

            if i > num_batches:
                plt.colorbar()
                plt.savefig(os.path.join(root, "latent_" + str(epoch) + ".png"))
                break


def plot_linear_interpolation(model, a, b, img_size, num, device):
    img = np.zeros((img_size, num * img_size))
    model.eval()
    with torch.no_grad():
        for i, alpha in enumerate(np.linspace(0, 1, num)):
            z = (alpha * torch.Tensor(a) + (1 - alpha) * torch.Tensor(b)).to(device)
            output = model.decoder(z)
            output = output.cpu().detach().numpy()
            img[..., i * img_size : (i + 1) * img_size] = output * 255

    img = Image.fromarray(img).convert("L")
    return img

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.optim import Adam
from model import VAE
from loss import ELBO
from tqdm import tqdm, trange
from utils import *



training_data = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

training_dataloader = DataLoader(training_data, batch_size=100, shuffle=True)

vae = VAE(latent_dim=2)
optimizer = Adam(vae.parameters())
loss_fn = ELBO(distribution="mse", latent_dim=2)


num_train_batches = len(training_dataloader)
epochs = 40
for e in trange(1,epochs+1, dynamic_ncols=True):
    vae.train()
    train_loss_rc = 0.
    train_loss_kl =0.
    train_loss = 0.
    for batch, (X,y) in enumerate(training_dataloader):
        z, mu, logsig, output = vae(X)
        loss, loss_rc, loss_kl=loss_fn(mu,logsig,output,X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loss_rc+=loss_rc
        train_loss_kl+=loss_kl

    train_loss /= num_train_batches
    train_loss_rc /= num_train_batches
    train_loss_kl /= num_train_batches
    tqdm.write(f"[TRAIN] Epoch: {e:3d}| Overall Loss: {train_loss:.2f}| Reconstruction Loss: {train_loss_rc:.2f}| KL Divergence: {train_loss_kl:.2f}")


    if e%10 ==0 and e!=0:
        random_sampling(vae,2,e,5,"./image")
        plot_latent_space(vae,e,training_dataloader,100,"./image")
        plot_manifold(vae,e,[-2,2],[-2,2],28,12,"./image")
        plot_linear_interpolation(vae,e,[-3,3],[3,-3],28,25,"./image")
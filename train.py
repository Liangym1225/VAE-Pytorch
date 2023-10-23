import argparse
from config import VAEConfig
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.optim import Adam
from model import VAE
from loss import ELBO
from tqdm import tqdm, trange
from utils import *
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str,default="MNIST")
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--distribution',type=str,default='gaussian')
parser.add_argument('-lr','--learning_rate',type=float,default=1e-3)
parser.add_argument('--latent_dimension',type=int,default=2)
parser.add_argument('--logging',type=str,default="None")
args=parser.parse_args()

config={
    "epochs":args.epochs,
    "batch_size":args.batch_size,
    "distribution":args.distribution,
    "learning_rate":args.learning_rate,
    "latent_dimension":args.latent_dimension,
    "logging":args.logging
}

config = VAEConfig(**config)
config.set_timestamp()
config.save_config()

if config.logging == "wandb":
    run = wandb.init(
        project="VAE",
        dir=config.get_base_dir(),
        config = config
    )

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
 
training_data = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
training_dataloader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True)

vae = VAE(latent_dim=config.latent_dimension).to(device)
optimizer = Adam(vae.parameters(),lr=config.learning_rate)
loss_fn = ELBO(distribution=config.distribution, latent_dim=config.latent_dimension)

num_train_batches = len(training_dataloader)
epochs = config.epochs
for e in trange(1,epochs+1, dynamic_ncols=True):
    vae.train()
    train_loss_rc = 0.
    train_loss_kl =0.
    train_loss = 0.
    for batch, (X,y) in enumerate(training_dataloader):
        X = X.to(device)
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
    if config.logging == "wandb":
        run.log({
            "Train/Overall Loss":train_loss,
            "Train/Reconstruction Loss":train_loss_rc,
            "Train/KL Divergence":train_loss_kl
        })

    """
    if e%10 ==0 and e!=0:
        random_sampling(vae,2,e,5,"./image")
        plot_latent_space(vae,e,training_dataloader,100,"./image")
        plot_manifold(vae,e,[-2,2],[-2,2],28,12,"./image")
        plot_linear_interpolation(vae,e,[-3,3],[3,-3],28,25,"./image")
    """

run.finish()
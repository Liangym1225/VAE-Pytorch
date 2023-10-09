import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def random_sampling(model, latent_dim, epoch, num, root):
    z = torch.randn(num**2,latent_dim)
    model.eval()
    with torch.no_grad():
        output = model.decoder(z)
    
    output.detach().numpy()
    output = np.squeeze(output)
    plt.figure()
    fig,ax = plt.subplots(nrows=5,ncols=5,figsize=(10,10))
    plt.gray()
    for i in range(num**2):
        idx = divmod(i,5)
        ax[idx].imshow(output[i])
        ax[idx].axis('off')
    
    fig.savefig(os.path.join(root,"samples_"+str(epoch)+".png"))

def plot_manifold(model,epoch, z0, z1, img_size, num, root):
    img = np.zeros((num*img_size, num*img_size))
    with torch.no_grad():
        for i, y in enumerate(np.linspace(z1[0],z1[1],num)):
            for j, x in enumerate(np.linspace(z0[0],z0[1],num)):
                z = torch.Tensor(([[x,y]]))
                output = model.decoder(z)
                output = output.detach().numpy()
                img[(num-1-i)*img_size:(num-i)*img_size, j*img_size:(j+1)*img_size]=output*255
    
    img = Image.fromarray(img).convert("L")
    img.save(os.path.join(root,"manifold_"+str(epoch)+".png"))


def plot_latent_space(model,epoch, dataloader, num_batches, root):
    plt.figure()
    model.eval()
    with torch.no_grad():
        for i, (X,y) in enumerate(dataloader):
            z,_,_,_=model(X)
            z=z.detach().detach().numpy()
            y=y.tolist()
            plt.scatter(z[:,0],z[:,1],c=y,cmap='tab10')

            if i>num_batches:
                plt.colorbar()
                plt.savefig(os.path.join(root,"latent_"+str(epoch)+".png"))
                break
            
def plot_linear_interpolation(model, epoch, a,b,img_size, num, root):
    img = np.zeros((img_size, num*img_size))
    with torch.no_grad():
        for i, alpha in enumerate(np.linspace(0,1,num)):
            z = alpha*torch.Tensor(a)+(1-alpha)*torch.Tensor(b)
            output = model.decoder(z)
            output = output.detach().numpy()
            img[...,i*img_size:(i+1)*img_size]=output*255
    
    img = Image.fromarray(img).convert("L")
    img.save(os.path.join(root, "interpolation_"+str(epoch)+".png"))







from fastai.basic_train import Learner
from fastai.basic_data import DataBunch, DatasetType
from fastai.basic_train import get_preds
from fastai.callback import CallbackHandler

from fastai_autoencoder.callback import ReplaceTargetCallback, VAEHook

import torch.nn as nn
import torch
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class AutoEncoderLearner(Learner):
    def __init__(self,data:DataBunch,rec_loss,enc:nn.Module,bn:nn.Module,dec:nn.Module,**kwargs):
        self.enc = enc
        self.bn = bn
        self.dec = dec
        self.rec_loss = rec_loss
        self.encode = nn.Sequential(enc,bn)
        
        self.inferring = False
        
        ae = nn.Sequential(enc,bn,dec)
        
        super().__init__(data, ae, loss_func=self.loss_func, **kwargs)
        
        # Callback to replace y with x during the training loop
        replace_cb = ReplaceTargetCallback(self)
        self.callbacks.append(replace_cb)
        
    def loss_func(self,x,x_rec,**kwargs):
        bs = x.shape[0]
        l = self.rec_loss(x, x_rec, reduction='none').view(bs, -1).sum(dim=-1).mean()
        return l
    
    def decode(self,z):
        x_rec = self.dec(z)
        return x_rec
    
    def plot_2d_latents(self,ds_type:DatasetType=DatasetType.Valid, n_batch = 10):
        """Plot a 2D map of latents colored by class"""
        z,y = self.get_latents(ds_type = ds_type, n_batch = n_batch)
        z,y = z.numpy(), y.numpy()
        
        print("Computing the TSNE projection")
        zs = TSNE(n_components=2).fit_transform(z)

        plt.figure(figsize = (16,12))
        plt.scatter(zs[:,0],z[:,1],c = y)
        plt.title("TSNE projection of latents on two dimensions")
        plt.show()
        
    def plot_rec(self,x=None,i=0,sz = 64,gray = True):
        """Plot a """
        self.model.cpu()
        if not isinstance(x,torch.Tensor):
            x,y = self.data.one_batch()
            
        x_rec = self.model(x)

        x_rec = x_rec[i].squeeze(1)
        x = x[i].squeeze(1)

        img = x.permute(1,2,0).numpy()
        img_r = x_rec.permute(1,2,0).detach().numpy()
        
        if gray:
            img = np.concatenate((img,)*3,axis = -1)
            img_r = np.concatenate((img_r,)*3,axis = -1)

        fig,ax = plt.subplots(2,figsize = (16,16))
        ax[0].imshow(img,cmap = "gray")
        ax[0].set_title("Original")

        ax[1].imshow(img_r,cmap = "gray")
        ax[1].set_title("Reconstruction")

        self.model.cuda()

    # We perturb the latent variables on each variable with different magnitudes 
    def plot_shades(self,x=None,i=0,n_perturb = 13, mag_perturb = 3, ax = None):
        """Plot the reconstruction of the """
        self.model.cpu()
        if not isinstance(x,torch.Tensor):
            x,y = self.data.one_batch()

        x = x[i].unsqueeze(0)

        # We get the latent code
        z = self.encode(x)
        n_z = z.shape[1]

        # We create a scale of perturbations
        mag_perturb = np.abs(mag_perturb)
        scale_perturb = np.linspace(-mag_perturb,mag_perturb,n_perturb)
        
        if not ax:
            fig, ax = plt.subplots(n_z,n_perturb, figsize=(16,12))
            fig.tight_layout()

        for i in range(n_z):
            for (j,perturb) in enumerate(scale_perturb):
                # For each z_i we add a minor perturbation
                z_perturb = z.clone()
                z_perturb[0,i] += perturb

                # We reconstruct our image
                x_rec = self.decode(z_perturb)

                # We plot it in the grid
                img = x_rec.squeeze(0).permute(1,2,0).detach().numpy()
                ax[i][j].set_axis_off()
                ax[i][j].imshow(img)
                ax[i][j].set_title(f"z_{i} with {round(perturb * 1e2) / 1e2}")
                
        self.model.cuda()
        
    def get_latents(self, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None,
                  with_loss:bool=False, n_batch=None, pbar=None):
        "Return predictions and targets on `ds_type` dataset."
        lf = self.loss_func if with_loss else None
        self.inferring = True
        output = get_preds(self.encode, self.dl(ds_type), cb_handler=CallbackHandler(self.callbacks),
                         activ=activ, loss_func=lf, n_batch=n_batch, pbar=pbar)
        self.inferring = False
        return output
    
    def get_error(self, ds_type:DatasetType=DatasetType.Valid,activ:nn.Module=None, n_batch=None, pbar=None):
        "Return predictions and targets on `ds_type` dataset."
        
        x_rec,x = get_preds(self.model, self.dl(ds_type), cb_handler=CallbackHandler(self.callbacks),
                         activ=activ, loss_func=None, n_batch=n_batch, pbar=pbar)
        loss_func = lambda x_rec,x : self.rec_loss(x, x_rec, reduction='none').view(x.shape[0], -1).sum(dim=-1)
        l = loss_func(x_rec,x)
        return l
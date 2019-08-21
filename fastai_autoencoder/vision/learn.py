from fastai_autoencoder.learn import AutoEncoderLearner

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
        
from fastai.callbacks import Hooks

def gray_to_rgb(img):
    return np.concatenate((img,)*3,axis=-1)

def register_output(m,i,o):
    return o

class VisionAELearner(AutoEncoderLearner):
    def plot_rec(self,x=None,random=False,n=5,gray = True,return_fig=False):
        """Plot a """
        self.model.cpu()
        if not isinstance(x,torch.Tensor):
            x,y = self.data.one_batch()
        
        bs = self.data.batch_size
        
        assert n < bs, f"Number of images to plot larger than batch size {n}>{bs}"
        
        if random:
            idx = np.random.choice(bs,n,replace = False)
        else:
            idx = np.arange(n)
        x = x[idx]
        
        x_rec = self.model(x)
    
        if isinstance(self.rec_loss,nn.CrossEntropyLoss):
            x_rec = x_rec.argmax(dim = 1,keepdim = True)
        
        fig,ax = plt.subplots(n,2,figsize = (20,20))
        
        for i in range(n):
            img = x[i].permute(1,2,0).numpy()
            img_r = x_rec[i].permute(1,2,0).detach().numpy()
    
            if gray:
                img = np.concatenate((img,)*3,axis = -1)
                img_r = np.concatenate((img_r,)*3,axis = -1)
                
            ax[i][0].imshow(img,cmap = "gray")
            ax[i][0].set_title("Original")
    
            ax[i][1].imshow(img_r,cmap = "gray")
            ax[i][1].set_title("Reconstruction")
        
        self.model.cuda()
        if return_fig:
            return fig
        
    def set_dec_modules(self,dec_modules:list):
        self.dec_modules = dec_modules

    def plot_rec_steps(self,x=None,random=False,n=5,gray = True,return_fig=False):
        """Plot a """
        self.model.cpu()
        if not isinstance(x,torch.Tensor):
            x,y = self.data.one_batch()
        
        bs = self.data.batch_size
        
        assert n < bs, f"Number of images to plot larger than batch size {n}>{bs}"
        
        if random:
            idx = np.random.choice(bs,n,replace = False)
        else:
            idx = np.arange(n)
        x = x[idx]
        
        modules = self.dec_modules
        
        with Hooks(modules,register_output) as hooks:
            x_rec = self.model(x)
            outputs = hooks.stored
            
            n_layers = len(outputs)
        
            fig,ax = plt.subplots(n,n_layers+1,figsize = (20,20))

            for i in range(n):
                img = x[i].permute(1,2,0).numpy()
                if gray : img = gray_to_rgb(img)
                
                ax[i][0].imshow(img,cmap = "gray")
                ax[i][0].set_title("Original")
                
                for j in range(n_layers):
                    img_r = outputs[j][i].mean(dim=0,keepdim=True).permute(1,2,0).detach().numpy()
                    if gray : img_r = gray_to_rgb(img_r)
                
                    ax[i][j+1].imshow(img_r,cmap = "gray")
                    ax[i][j+1].set_title(f"Step {j}")
            self.model.cuda()
            if return_fig:
                return fig
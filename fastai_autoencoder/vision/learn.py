from fastai_autoencoder.learn import AutoEncoderLearner

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class VisionAELearner(AutoEncoderLearner):
    def plot_rec(self,x=None,n=5,gray = True,return_fig=False):
        """Plot a """
        self.model.cpu()
        if not isinstance(x,torch.Tensor):
            x,y = self.data.one_batch()
        
        bs = self.data.batch_size
        
        assert n < bs, f"Number of images to plot larger than batch size {n}>{bs}"
        
        idx = np.random.choice(bs,n,replace = False)
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
            
        if return_fig:
            return fig
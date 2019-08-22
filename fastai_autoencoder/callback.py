from fastai.callbacks import LearnerCallback
from fastai.basic_train import Learner
from fastai.callbacks.hooks import HookCallback
from fastai_autoencoder.bottleneck import VAELinear
import torch.nn as nn
import torch
import numpy as np

def get_layer(m,buffer,layer):
    """Function which takes a list and a model append the elements"""
    for c in m.children():
        if isinstance(c,layer):
            buffer.append(c)
        get_layer(c,buffer,layer)

class ReplaceTargetCallback(LearnerCallback):
    """Callback to modify the loss of the learner to compute the loss against x"""
    _order = 9999
    
    def __init__(self,learn:Learner):
        super().__init__(learn)
        
    def on_batch_begin(self,last_input,last_target,train,**kwargs):
        # We keep the original x to compute the reconstruction loss
        if not self.learn.inferring:
            return {"last_input" : last_input,"last_target" : last_input}
        else:
            return {"last_input" : last_input,"last_target" : last_target} 
        
class VQVAEHook(HookCallback):
    """Callback to modify the loss of the learner to compute the loss against x"""
    _order = 10000
    
    def __init__(self, learn,beta = 1,do_remove:bool=True):
        super().__init__(learn)
        
        # We look for the VAE bottleneck layer
        self.learn = learn
        
        self.loss = []
        
        buffer = []
        get_layer(learn.model,buffer,VQVAEBottleneck)
        if not buffer:
            raise NotImplementedError("No VQ VAE Bottleneck found")
            
        self.modules = buffer
        self.do_remove = do_remove
        
    def on_backward_begin(self,last_loss,**kwargs):
        total_loss = last_loss + self.current_loss
        
        return {"last_loss" : total_loss}
    
    def hook(self, m:nn.Module, i, o):
        "Save the latents of the bottleneck"
        self.current_loss = m.loss
        self.loss.append(m.loss)
        
class VAEHook(HookCallback):
    """Hook to register the parameters of the latents during the forward pass to compute the KL term of the VAE"""
    
    def __init__(self, learn,beta = 1,do_remove:bool=True):
        super().__init__(learn)
        
        # We look for the VAE bottleneck layer
        self.learn = learn
        self.beta = beta
        
        self.loss = []
        
        buffer = []
        get_layer(learn.model,buffer,VAELinear)
        if not buffer:
            raise NotImplementedError("No Bayesian Linear found")
            
        self.modules = buffer
        self.do_remove = do_remove
    
    def on_backward_begin(self,last_loss,**kwargs):
        n = len(self.learn.mu)
        # We add the KL term of each Bayesian Linear Layer
        
        
        mu = self.learn.mu
        logvar = self.learn.logvar
        kl = self.beta * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean())
        
        total_loss = last_loss + kl
        self.loss.append({"rec_loss":last_loss.cpu().detach().numpy(),"kl_loss":kl.cpu().detach().numpy(),
                          "total_loss":total_loss.cpu().detach().numpy()})
        
        return {"last_loss" : total_loss}
        
    def hook(self, m:nn.Module, i, o):
        "Save the latents of the bottleneck"
        mu,logvar = o
        self.learn.mu = mu
        self.learn.logvar = logvar

class HighFrequencyLoss(LearnerCallback):
    def __init__(self, learn,low_ratio = 0.15,threshold = 1e-2,scaling=True,debug=False):
        super().__init__(learn)
        
        # We look for the VAE bottleneck layer
        assert low_ratio < 0.5, "Low ratio too high"
        self.low_ratio = low_ratio
        self.window_size = int(28 * low_ratio)
        self.threshold = threshold
        self.scaling = scaling
        self.debug = debug
        
    def get_exponent(self,x): return np.floor(np.log10(np.abs(x))).astype(int)
        
    def on_backward_begin(self,last_loss,**kwargs):
        
        x = kwargs["last_input"]
        x_rec = kwargs["last_output"]
        
        # First we get the fft of the batch
        f = np.fft.fft2(x.squeeze(1))
        fshift = np.fft.fftshift(f,axes=(1,2))

        # We zero the low frequencies
        rows, cols = x.shape[-2], x.shape[-1]
        crow,ccol = rows//2 , cols//2
        fshift[:,crow-self.window_size:crow+self.window_size, ccol-self.window_size:ccol+self.window_size] = 0
        
        # We reconstruct the image
        f_ishift = np.fft.ifftshift(fshift,axes=(1,2))
        img_back = np.fft.ifft2(f_ishift)
        
        # We keep the indexes of pixels with high values
        img_back = np.abs(img_back)
        img_back = img_back / img_back.sum(axis=(1,2),keepdims=True)
        idx = (img_back > self.threshold)
        
        img_back = torch.tensor(img_back[idx]).cuda()
        mask = torch.ByteTensor(idx).cuda()
        
        # We select only the pixels with high values
        x_hf = torch.masked_select(x.view_as(mask),mask)
        x_rec_hf = torch.masked_select(x_rec.view_as(mask),mask)
        
        bs = x.shape[0]
        diff = img_back * self.learn.rec_loss(x_hf,x_rec_hf)
        
        hf_loss = diff.sum() / bs
        
        # If we scale it we put both losses on the same scale
        if self.scaling:
            rescale_factor = 10**(self.get_exponent(last_loss.item()) - self.get_exponent(hf_loss.item()))
            hf_loss *= rescale_factor
            
        total_loss = last_loss + hf_loss
        
        output = {"last_loss" : total_loss}
        if self.debug:
            print(f"Using High Frequency Loss")
            print(f"Loss before : {last_loss}")
            print(f"High frequency loss : {hf_loss}")
            print(output)
        return output
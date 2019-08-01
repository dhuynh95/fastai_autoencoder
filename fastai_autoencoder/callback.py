from fastai.callbacks import LearnerCallback
from fastai.basic_train import Learner
from fastai.callbacks.hooks import HookCallback
from fastai_autoencoder.model import VAELinear, BayesianLinear
import torch.nn as nn

def get_layer(m,buffer,layer):
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

class VAEHook(HookCallback):
    """Hook to register the parameters of the latents during the forward pass to compute the KL term of the VAE"""
    
    def __init__(self, learn:Learner,beta = 1,do_remove:bool=True):
        super().__init__(learn)
        
        # We look for the VAE bottleneck layer
        self.learn = learn
        self.beta = beta
        self.kl = []
        self.l = []
        buffer = []
        get_layer(learn.model,buffer,VAELinear)
        if not buffer:
            raise NotImplementedError("No Bayesian Linear found")
        self.modules = buffer
        self.do_remove = do_remove
        
    def on_batch_begin(self,**kwargs):
        # We set the 
        self.learn.mu = []
        self.learn.logvar = []
    
    def on_backward_begin(self,last_loss,**kwargs):
        n = len(self.learn.mu)
        # We add the KL term of each Bayesian Linear Layer
        self.l.append(last_loss)
        kl = 0
        for i in range(n):
            mu = self.learn.mu[i]
            logvar = self.learn.logvar[i]
            kl += self.beta * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean())
        self.kl.append(kl)
        last_loss += kl
        return {"last_loss" : last_loss}
        
    def hook(self, m:nn.Module, i, o):
        "Save the latents of the bottleneck"
        mu,logvar = o
        self.learn.mu.append(mu)
        self.learn.logvar.append(logvar)
        
class BayesianHook(HookCallback):
    """Hook to register the parameters of the latents during the forward pass to compute the KL term of the VAE"""
    
    def __init__(self, learn:Learner,beta = 1,do_remove:bool=True):
        super().__init__(learn)
        
        # We look for the VAE bottleneck layer
        self.learn = learn
        self.beta = beta
        self.kl = []
        self.l = []
        buffer = []
        get_layer(learn.model,buffer,BayesianLinear)
        if not buffer:
            raise NotImplementedError("No Bayesian Linear found")
        self.modules = buffer
        self.do_remove = do_remove
        
    def on_batch_begin(self,**kwargs):
        # We set the 
        self.learn.w_mu = []
        self.learn.w_logvar = []
        
        self.learn.b_mu = []
        self.learn.b_logvar = []
    
    def on_backward_begin(self,last_loss,**kwargs):
        n = len(self.learn.w_mu)
        # We add the KL term of each Bayesian Linear Layer
        self.l.append(last_loss)
        kl = 0
        for i in range(n):
            w_mu,w_logvar = self.learn.w_mu[i], self.learn.w_logvar[i]
            b_mu,b_logvar = self.learn.b_mu[i], self.learn.b_logvar[i]
            
            kl += self.beta * (-0.5 * (1 + w_logvar - w_mu.pow(2) - w_logvar.exp()).sum(dim=-1).mean())
            kl += self.beta * (-0.5 * (1 + b_logvar - b_mu.pow(2) - b_logvar.exp()).sum(dim=-1).mean())
        self.kl.append(kl)
        last_loss += kl
        return {"last_loss" : last_loss}
        
    def hook(self, m:nn.Module, i, o):
        "Save the latents of the bottleneck"
        w_mu,w_logvar = m.w_mu, m.w_logvar
        b_mu,b_logvar = m.b_mu, m.b_logvar
        
        self.learn.w_mu.append(w_mu)
        self.learn.w_logvar.append(w_logvar)
        
        self.learn.b_mu.append(b_mu)
        self.learn.b_logvar.append(b_logvar)
        

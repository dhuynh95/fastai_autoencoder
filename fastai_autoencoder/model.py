import torch.nn as nn
import torch
import torch.nn.functional as F

class NormalSampler(nn.Module):
    def __init__(self):
        super(NormalSampler,self).__init__()
    
    def forward(self,mu,logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

class VAELinear(nn.Module):
    def __init__(self,in_features,out_features):
        super(VAELinear,self).__init__()
        self.linear = nn.Linear(in_features,out_features)
        self.mu_linear = nn.Linear(out_features,out_features)
        self.logvar_linear = nn.Linear(out_features,out_features)

    def forward(self,x):
        x = self.linear(x)
        mu = self.mu_linear(x)
        logvar = self.logvar_linear(x)

        return mu,logvar

class VAELayer(nn.Module):
    def __init__(self,in_features,out_features):
        super(VAELayer,self).__init__()
        self.blinear = VAELinear(in_features,out_features)
        self.sampler = NormalSampler()

    def forward(self,x):
        mu,logvar = self.blinear(x)
        z = self.sampler(mu,logvar)
        return z
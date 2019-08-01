import torch.nn as nn
import torch
from fastai_autoencoder.model import VAELayer

class VAEBottleneck(nn.Module):
    def __init__(self,nfs:list,activation):
        super(VAEBottleneck,self).__init__()

        n = len(nfs)
        layers = [nn.Linear(in_features=nfs[i], out_features=nfs[i+1]) if i < n -2 
        else VAELayer(in_features=nfs[i], out_features=nfs[i+1]) for i in range(n-1)]

        self.fc = nn.Sequential(*layers)
    
    def forward(self,x):
        bs = x.size(0)
        x = x.view(bs, -1)
        z = self.fc(x)
        
        return z

class Bottleneck(nn.Module):
    def __init__(self,nfs,activation):
        super(Bottleneck,self).__init__()

        n = len(nfs)
        layers = [nn.Linear(in_features=nfs[i], out_features=nfs[i+1]) for i in range(n-1)]

        self.fc = nn.Sequential(*layers)
    
    def forward(self,x):
        bs = x.size(0)
        x = x.view(bs, -1)
        z = self.fc(x)
        
        return z
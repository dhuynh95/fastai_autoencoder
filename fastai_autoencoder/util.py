import torch
import torch.nn as nn
from fastai.torch_core import requires_grad
from fastai.callbacks import Hooks
from fastai.torch_core import flatten_model

class DepthwiseConv(nn.Module):
    def __init__(self,ni,activation,ks=3,stride=1,conv = nn.Conv2d,bn = None):
        super(DepthwiseConv,self).__init__()
        if bn:
            self.conv = conv(ni,ni,kernel_size=ks,padding=ks //2,stride = stride,groups = ni, bias = False)
            self.bn = bn(ni)
        else:
            self.conv = conv(ni,ni,kernel_size=ks,padding=ks //2,stride = stride,groups = ni, bias = True)
            self.bn = None
        self.act_fn = activation
    
    def forward(self,x):
        output = self.act_fn(self.conv(x)) if not self.bn else self.act_fn(self.bn(self.conv(x)))
        return output
    
    @property
    def bias(self): return self.conv.bias.data if not self.bn else self.bn.bias.data
    
    @bias.setter
    def bias(self,v):
        if not self.bn:
            self.conv.bias.data = v
        else:
            self.bn.bias.data = v
    
    @property
    def weight(self): return self.conv.weight

class PointwiseConv(nn.Module):
    def __init__(self,ni,nf,activation,conv = nn.Conv2d,bn = nn.BatchNorm2d):
        super(PointwiseConv,self).__init__()
        self.conv = conv(ni,nf,kernel_size=1,stride=1,padding=0,bias = False)
        self.bn = bn(nf)
        self.act_fn = activation
        
    def forward(self,x):
        output = self.act_fn(self.bn(self.conv(x)))
        return output
    
    @property
    def bias(self): return self.bn.bias
    
    @bias.setter
    def bias(self,v) : self.bn.bias = v
    
    @property
    def weight(self): return self.conv.weight

class MobileConv(nn.Module):
    def __init__(self,ni,nf,activation,ks=3,stride = 1,conv = nn.Conv2d,bn = nn.BatchNorm2d):
        super(MobileConv,self).__init__()
        self.depth_conv = DepthwiseConv(ni,activation,ks,stride,conv,bn)
        self.point_conv = PointwiseConv(ni,nf,activation,conv,bn)
    
    def forward(self,x):
        output = self.point_conv(self.depth_conv(x))
        return output

def register_stats(m,i,o):
    d = o.data
    mean, std = d.mean().item(),d.std().item()
    output = {"mean":mean,"std":std,"m":m}
    return output

def lsuv_init(model,xb,layers=None,types = (nn.Linear,nn.Conv2d),unfreeze = True):
    """LSUV init of the model.
    """
    output = []
    
    # If no layers are specified we will grab the CNN and Linear layers
    if not layers:
        layers = flatten_model(model)
        layers = [layer for layer in layers if isinstance(layer,types) ]
    
    requires_grad(model,False)
    print("Freezing all layers")
    with Hooks(layers,register_stats) as hooks:
        for i,hook in enumerate(hooks):
            # We first get the module
            model(xb)
            m = hook.stored["m"]
            
            while model(xb) is not None and abs(hook.stored["mean"])  > 1e-3: m.bias.data -= hook.stored["mean"]
            while model(xb) is not None and abs(hook.stored["std"]-1) > 1e-3: m.weight.data /= hook.stored["std"]
            
            output.append(f"Layer {i} named {str(m)} with m={hook.stored['mean']},std={hook.stored['std']}")
    
    if unfreeze:
        print("Unfreezing all layers")
        requires_grad(model,True)
    return output
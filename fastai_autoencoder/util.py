import torch
import torch.nn as nn
from fastai.torch_core import requires_grad
from fastai.callbacks import Hooks
from fastai.torch_core import flatten_model

class DepthwiseConv(nn.Module):

    def __init__(self,in_channels,kernel_size=3,stride=1,bias=True,conv = nn.Conv2d,**kwargs):
        super(DepthwiseConv,self).__init__()
        self.conv = conv(in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_size,
                         padding=kernel_size //2,stride = stride,groups = in_channels, bias = bias)

    def forward(self,x):
        output = self.conv(x)
        return output

class PointwiseConv(nn.Module):

    def __init__(self,in_channels,out_channels,bias = True,conv = nn.Conv2d,**kwargs):
        super(PointwiseConv,self).__init__()
        self.conv = conv(in_channels=in_channels,out_channels=out_channels,kernel_size=1,
                         stride=1,padding=0,bias = bias)

    def forward(self,x):
        output = self.conv(x)
        return output

class MobileConv(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=3,stride = 1,bias = True,conv = nn.Conv2d,
                dw_conv = DepthwiseConv, pw_conv = PointwiseConv,**kwargs):
        super(MobileConv,self).__init__()
        self.depth_conv = dw_conv(in_channels,kernel_size,stride,bias,conv)
        self.point_conv = pw_conv(in_channels,out_channels,bias,conv,)

    def forward(self,x):
        output = self.point_conv(self.depth_conv(x))
        return output

def zero_bias(conv_layer,conv_type = nn.Conv2d):
    for c in conv_layer.children():
        if isinstance(c,conv_type):
            c.bias.requires_grad = False
            c.bias.data = torch.zeros_like(c.bias)
        else:
            zero_bias(c)

class ConvBnRelu(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size = 3, stride = 1, bias = True, conv = nn.Conv2d,
                 bn = nn.BatchNorm2d, act_fn = nn.ReLU,**kwargs):
        super(ConvBnRelu,self).__init__()
        self.conv = conv(in_channels=in_channels,out_channels=out_channels,
                         kernel_size=kernel_size,stride = stride, bias=bias)

        if bn:
            self.bn = bn(out_channels)
            # If there is a bn we remove the bias term of the Conv
            zero_bias(self.conv)
        if act_fn:
            self.act_fn = act_fn(inplace = True)
    def forward(self,x):
        output = self.conv(x)
        if self.bn:
            output = self.bn(output)
        if self.act_fn:
            output = self.act_fn(output)
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
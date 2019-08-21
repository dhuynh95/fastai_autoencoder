import torch.nn as nn
import torch
from fastai_autoencoder.util import *
from fastai_autoencoder.decoder import SpatialDecoder2D

class ResBlock(nn.Module):
    def __init__(self,in_channels,kernel_size=3,stride=1,padding=1,conv = ConvBnRelu,**kwargs):
        super(ResBlock,self).__init__()
        self.conv1 = conv(in_channels=in_channels,out_channels=in_channels//2,kernel_size=kernel_size,
                        stride=stride,padding=padding,**kwargs)
        self.conv2 = conv(in_channels=in_channels//2,out_channels=in_channels,kernel_size=kernel_size,
                        stride=stride,padding=padding,**kwargs)
        
    def forward(self,x):
        h = self.conv2(self.conv1(x))
        return x + h
    
class DenseBlock(nn.Module):
    def __init__(self,in_channels,kernel_size=3,stride=1,padding=1,conv = ConvBnRelu,**kwargs):
        super(DenseBlock,self).__init__()
        self.conv = ResBlock(in_channels,kernel_size=kernel_size,stride=stride,padding=padding,conv=conv,**kwargs)
        
    def forward(self,x):
        h = self.conv(x)
        output = torch.cat([x,h],dim=1)
        return output

def create_encoder(nfs,ks,conv=nn.Conv2d,bn=nn.BatchNorm2d,act_fn = nn.ReLU):
    n = len(nfs)
    
    conv_layers = [nn.Sequential(ConvBnRelu(nfs[i],nfs[i+1],kernel_size=ks[i],
                                            conv = conv,bn=bn,act_fn=act_fn,padding = ks[i] //2),
                                 Downsample(channels=nfs[i+1],filt_size=3,stride=2))
                                   for i in range(n-1)]        
    convs = nn.Sequential(*conv_layers)
    
    return convs

def create_encoder_denseblock(n_dense,c_start):
    first_layer = nn.Sequential(ConvBnRelu(1,c_start,kernel_size=3,padding = 1),
                                ResBlock(c_start),
                                Downsample(channels=4,filt_size=3,stride=2))
    
    layers = [first_layer] + [
        nn.Sequential(
            DenseBlock(c_start * (2**c)),
            Downsample(channels=c_start * (2**(c+1)),filt_size=3,stride=2)) for c in range(n_dense)
    ]
    
    model = nn.Sequential(*layers)
    
    return model

def create_decoder(nfs,ks,size,conv = nn.Conv2d,bn = nn.BatchNorm2d,act_fn = nn.ReLU):
    n = len(nfs)
    
    # We add two channels to the first layer to include x and y channels
    first_layer = ConvBnRelu(nfs[0] + 2, nfs[1],conv = PointwiseConv,bn=bn,act_fn=act_fn)

    conv_layers = [first_layer] + [ConvBnRelu(nfs[i],nfs[i+1],kernel_size=ks[i-1],
                                              padding = ks[i-1] // 2,conv = conv,bn=bn,act_fn=act_fn)
                                   for i in range(1,n - 1)]        
    dec_convs = nn.Sequential(*conv_layers)
    
    dec = nn.Sequential(SpatialDecoder2D(size),dec_convs)
    
    return dec
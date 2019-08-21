import torch.nn as nn
import torch
from fastai_autoencoder.util import PointwiseConv

class SpatialDecoder2D(nn.Module):
    def __init__(self,size):
        super(SpatialDecoder2D,self).__init__()
        
        # Coordinates for the broadcast decoder
        self.size = size
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)
        
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
        
    def forward(self,z):
        batch_size = z.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        # Shape: NxDx1x1
        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.size, self.size)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)

        return x

class SpatialDecoder1D(nn.Module):
    def __init__(self,cnn,nfs,activation,im_size):
        super(Decoder1D,self).__init__()
        self.cnn = cnn
        
        # Coordinates for the broadcast decoder
        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))

        n = len(nfs)
        # We add two channels to the first layer to include x and y channels
        first_layer = PointwiseConv(nfs[0]+2,nfs[1],activation,conv = cnn)

        conv_layers = [first_layer] + [cnn(nfs[i],nfs[i+1],activation, 3) for i in range(1,n - 1)]        
        self.dec_convs = nn.Sequential(*conv_layers)
        
    def forward(self,z):
        batch_size = z.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        # Shape: NxDx1x1
        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1), z), dim=1)

        x = self.dec_convs(x)

        return x
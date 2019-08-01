import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, cnn, nfs,activation):
        super(Encoder, self).__init__()
        
        n = len(nfs)
        conv_layers = [cnn(nfs[i],nfs[i+1],activation, 3,stride = 2) for i in range(n - 1)]        
        self.enc_convs = nn.Sequential(*conv_layers)
        
    def forward(self,x):
        x = self.enc_convs(x)
        
        return x
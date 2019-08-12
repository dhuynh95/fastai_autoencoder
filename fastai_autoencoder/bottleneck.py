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

class VQVAEBottleneck(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VQVAEBottleneck, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

        self.inferring = False
        
    def forward(self, inputs):
        # convert inputs from BxCxHxW to BxHxWxC

        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        
        # Store the BxHxWxC shape
        input_shape = inputs.shape
        
        # Flatten input from BxHxWxC to BHWxC
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances between our input BHWxC and the embeddings KxC which outputs a BHWxK matrix
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Computing the closest BHWxK to BHWx1
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # If we are on inferrence mode we just output the discrete encoding in shape BxHxW
        if self.inferring:
            return encoding_indices.view(input_shape[:-1])
        
        # We reshape from BHWxC to BxHxWxC
        quantized = self._embedding(encoding_indices).view(input_shape)
        
        # We compute the loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        self.loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # We switch the gradients
        quantized = inputs + (quantized - inputs).detach()
        
        # We permute from BxHxWxC to BxCxHxW
        quantized = quantized.permute(0,3,1,2).contiguous()
        
        return quantized

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
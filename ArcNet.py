import torch.nn.functional as F
import torchvision as tv
import torch.nn as nn
import torch

class ArcNet(nn.Module):
    def __init__(self,num_classes,latent_dim,s=20,m=0.1): # orig, s=10,m=0.1
        super().__init__()
        self.s = s # scale
        self.m = torch.tensor(m) # margin
        self.w=nn.Parameter(torch.rand(latent_dim,num_classes)) #2*10
    def forward(self, embedding):
        embedding = F.normalize(embedding,dim=1) # normalize latent output
        w = F.normalize(self.w,dim=0) # normalize weights of ArcCos network
        cos_theta = torch.matmul(embedding, w)/self.s # /10
        sin_theta = torch.sqrt(1.0-torch.pow(cos_theta,2))
        cos_theta_m = cos_theta*torch.cos(self.m) - sin_theta*torch.sin(self.m)
        cos_theta_scaled = torch.exp(cos_theta * self.s)
        sum_cos_theta = torch.sum(torch.exp(cos_theta*self.s),dim=1,keepdim=True) - cos_theta_scaled 
        top = torch.exp(cos_theta_m*self.s)
        arcout = (top/(top + sum_cos_theta))
        return arcout
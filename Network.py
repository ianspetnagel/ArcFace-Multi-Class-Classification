import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
class Network(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            #nn.Conv2d(1, 64, 3, 2, 1), # MNIST is gray scale, so input channels=1
            nn.Conv2d(3, 64, 3, 2, 1), # CIFAR-10 uses input channels=3 so
            # this line and comment above line
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,16,3, 2, 1),
            nn.ReLU())
        self.linear_layer = nn.Linear(16*4*4,latent_dim)
        #self.output_layer = nn.Linear(latent_dim,10,bias=False)

    def forward(self, xs):
        cnn_out = self.cnn_layers(xs)
        flatten = cnn_out.reshape(-1,16*4*4)
        latent_out = self.linear_layer(flatten)
        #output = self.output_layer(latent_out)
        return latent_out
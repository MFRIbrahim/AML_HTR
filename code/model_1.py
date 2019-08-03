import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # ---CNN layers---
        conv_kernel = [5, 5, 3, 3, 3]
        channels = [1, 32, 64, 128, 128, 256]
        pool_kernel_stride = [(2,2), (2,2), (2,1), (2,1), (2,1)]
        
        for i in range(len(conv_kernel)):
            if conv_kernel[i] == 5:
                self.layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=conv_kernel[i], stride=1, padding=2))
            else:
                self.layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=conv_kernel[i], stride=1, padding=1))
            self.layers.append(nn.BatchNorm2d(num_features=channels[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=pool_kernel_stride[i], stride=pool_kernel_stride[i], padding=0))
        
        # ---RNN layers---
        
    
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
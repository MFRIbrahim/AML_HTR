import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

"""
kernelVals = [5, 5, 3, 3, 3]
featureVals = [1, 32, 64, 128, 128, 256]
strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]

IMAGE: 32x128

"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        
        # ---CNN layers---
        self.cnn_layers = nn.ModuleList()
        
        conv_kernel = [5, 5, 3, 3, 3]
        channels = [1, 32, 64, 128, 128, 256]
        pool_kernel_stride = [(2,2), (2,2), (2,1), (2,1), (2,1)]
        
        for i in range(len(conv_kernel)):
            if conv_kernel[i] == 5:
                self.cnn_layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=conv_kernel[i], stride=1, padding=2))
            else:
                self.cnn_layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=conv_kernel[i], stride=1, padding=1))
            self.cnn_layers.append(nn.BatchNorm2d(num_features=channels[i+1]))
            self.cnn_layers.append(nn.ReLU())
            self.cnn_layers.append(nn.MaxPool2d(kernel_size=pool_kernel_stride[i], stride=pool_kernel_stride[i], padding=0))
        
        # ---RNN layers---    
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
        #---CNN layer---
        self.cnn = nn.Conv2d(in_channels=512, out_channels=80, kernel_size=1, stride=1, padding=0)
    
    
    def forward(self, x):
        for layer in self.cnn_layers:
            #print(x.shape)
            #print(layer)
            x = layer(x)
        print("After CNN layers:", x.shape)
        #transform x for the lstm
        x = torch.squeeze(x)
        x = x.permute(0,2,1)
        print("Transformation for LSTM:", x.shape)
        x = self.lstm(x)
        x = x[0]
        print("After LSTM:", x.shape)
        x = torch.unsqueeze(x,2)
        x = x.permute(0,3,2,1)
        print("Transformation for last CNN layer:", x.shape)
        x = self.cnn(x)
        print("After last CNN layer:", x.shape)
        x = torch.squeeze(x)
        print("Transformation for CTC:", x.shape)
        return x
  

        
        
"""       
self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
self.batchnorm1 = nn.BatchNorm2d(num_features=32)
self.relu = nn.ReLU() 
self.maxpool1 = nn.Maxpool2d(kernel_size=(2,2), stride=(2,2))
"""         

net = Net()
#print(net)
a = torch.rand(50, 1, 32, 128)
print("Input:", a.shape)

b = net(a)
#print(b.shape)


"""
b = torch.squeeze(b)
print(b.shape)

b = b.permute(0,2,1)
print(b.shape)

"""
"""
def get_padding(size, kernel_size, stride, dilation):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) //2
    return padding

padding = get_padding(128, 3, 1, 1)
print(padding)"""

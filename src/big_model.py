import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, lstm_layers=2, bidirectional=True, dropout=0.0):
        super(Net, self).__init__()

        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        # ---CNN layers---
        self.cnn_layers = nn.ModuleList()


        self.cnn_layers.append(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=2))
        self.cnn_layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0))
        self.cnn_layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(1,1), padding=2))
        self.cnn_layers.append(nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0))
        self.cnn_layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1))
        self.cnn_layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0))
        self.cnn_layers.append(nn.BatchNorm2d(num_features=128))
        self.cnn_layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1))
        self.cnn_layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1))
        self.cnn_layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0))
        self.cnn_layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1))
        self.cnn_layers.append(nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0))
        self.cnn_layers.append(nn.BatchNorm2d(num_features=512))
        self.cnn_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1))
        self.cnn_layers.append(nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0))


        # ---LSTM---
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

        #---last CNN layer---
        self.cnn = nn.Conv2d(in_channels=1024, out_channels=80, kernel_size=(1,1), stride=(1,1), padding=0)

        self.hidden = ()


    def init_hidden(self, batch_size, device):
        self.hidden = (self.create_hidden_parameter(batch_size, device=device),
                       self.create_hidden_parameter(batch_size, device=device))

    def create_hidden_parameter(self, batch_size, device):
        tensor = torch.Tensor(self.lstm_layers * (self.bidirectional + 1), batch_size, 512).float()
        tensor = nn.init.xavier_uniform_(tensor).to(device)
        return nn.Parameter(tensor, requires_grad=True)


    def forward(self, x):
        # pass through CNN layers
        for layer in self.cnn_layers:
            x = layer(x)
        # transformation for LSTM
        x = x.squeeze(2)
        x = x.permute(0,2,1)
        # pass through LSTM
        x, self.hidden = self.lstm(x, self.hidden)
        # transformation for last CNN layer
        x = x.unsqueeze(2)
        x = x.permute(0,3,2,1)
        # pass through last CNN layer
        x = self.cnn(x)
        # transform for CTC_loss calc and text decoding
        x = x.squeeze(2)
        x = x.permute(2,0,1)
        return x

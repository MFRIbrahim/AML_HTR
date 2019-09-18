import torch
from torch import nn as nn


def get_model_by_name(name):
    if name == "Net":
        return lambda params: Net(**params)
    else:
        raise RuntimeError(f"Unknown specified network '{name}'")


class Net(nn.Module):
    def __init__(self, lstm_layers=2, bidirectional=True, dropout=0.0):
        super().__init__()

        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        # ---CNN layers---
        self.cnn_layers = nn.ModuleList()

        conv_kernel = [5, 5, 3, 3, 3]
        channels = [1, 32, 64, 128, 128, 256]
        pool_kernel_stride = [(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)]

        for i in range(len(conv_kernel)):
            padding = 2 if conv_kernel[i] == 5 else 1

            self.cnn_layers.append(nn.Conv2d(in_channels=channels[i],
                                             out_channels=channels[i + 1],
                                             kernel_size=conv_kernel[i],
                                             stride=1, padding=padding))

            self.cnn_layers.append(nn.BatchNorm2d(num_features=channels[i + 1]))
            self.cnn_layers.append(nn.ReLU())
            self.cnn_layers.append(nn.MaxPool2d(kernel_size=pool_kernel_stride[i],
                                                stride=pool_kernel_stride[i],
                                                padding=0))

        # ---LSTM---
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=256,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout)

        # ---last CNN layer---
        self.cnn = nn.Conv2d(in_channels=(bidirectional + 1) * 256,
                             out_channels=80,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.hidden = ()

    def init_hidden(self, batch_size, device):
        self.hidden = (self.create_hidden_parameter(batch_size, device=device),
                       self.create_hidden_parameter(batch_size, device=device))

    def create_hidden_parameter(self, batch_size, device):
        tensor = torch.Tensor(self.lstm_layers * (self.bidirectional + 1), batch_size, 256).float()
        tensor = nn.init.xavier_uniform_(tensor).to(device)
        return nn.Parameter(tensor, requires_grad=True)

    def forward(self, x):
        # pass through CNN layers
        for layer in self.cnn_layers:
            x = layer(x)
        # transformation for LSTM
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        # pass through LSTM
        x, self.hidden = self.lstm(x, self.hidden)
        # transformation for last CNN layer
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        # pass through last CNN layer
        x = self.cnn(x)
        # transform for CTC_loss calc and text decoding
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        return x

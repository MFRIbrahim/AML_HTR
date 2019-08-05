import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


CHAR_LIST = list(" !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
CHAR_DICT = {}
for i in range(len(CHAR_LIST)):
    CHAR_DICT[i] = CHAR_LIST[i]
    
def Decoder(matrix):
    # matrix with shape (seq_len, batch_size, num_of_characters) --> (32,50,80)
    C = np.argmax(matrix, axis=2).T  
    output = []
    for i in range(C.shape[0]):
        sub = []
        for j in range(C.shape[1]):
            sub.append(CHAR_DICT[C[i][j]])
        output.append(sub)
    return output    
    
    
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
        
        # ---LSTM---    
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
        #---last CNN layer---
        self.cnn = nn.Conv2d(in_channels=512, out_channels=80, kernel_size=1, stride=1, padding=0)
    
    
    def forward(self, x):
        # pass through CNN layers
        for layer in self.cnn_layers:
            x = layer(x)
        # transformation for LSTM
        x = torch.squeeze(x)
        x = x.permute(0,2,1)
        # pass through LSTM
        x = self.lstm(x)
        x = x[0]
        # transformation for last CNN layer
        x = torch.unsqueeze(x,2)
        x = x.permute(0,3,2,1)
        # pass through last CNN layer 
        x = self.cnn(x)
        # transform for CTC_loss calc and text decoding
        x = torch.squeeze(x)
        x = x.permute(2,0,1)
        return x


def training():
    net = Net()
    learning_rate = 0.001
    loss = nn.CTCLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


net = Net()

a = torch.rand(50, 1, 32, 128)


b = net(a)

ctc_input = F.log_softmax(b)
ctc_target = torch.randint(low=1, high=80, size=(50, 32), dtype=torch.long)
input_lengths = torch.full(size=(50,), fill_value=32, dtype=torch.long)
target_lengths = torch.randint(low=1, high=32, size=(50,), dtype=torch.long)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(ctc_input, ctc_target, input_lengths, target_lengths)
loss.backward()

print(loss)
c = b.data.numpy()
print(Decoder(c))

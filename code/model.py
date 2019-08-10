import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from PIL import Image as PImage
import torchvision


CHAR_LIST = list(" !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
CHAR_DICT = {}
for i in range(len(CHAR_LIST)):
    CHAR_DICT[i] = CHAR_LIST[i]
INV_CHAR_DICT = {v: k for k, v in CHAR_DICT.items()}

def Decoder(matrix):
    # matrix with shape (seq_len, batch_size, num_of_characters) --> (32,50,80)
    C = np.argmax(matrix, axis=2)
    output = []
    #iterate over dim 1 first, since those are the batches
    for i in range(C.shape[1]):
        sub = []
        #iterate over the sequence
        for j in range(C.shape[0]):
            sub.append(CHAR_DICT[C[j][i]])
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

def encodeWord(Y):
    new_Y = []
    for w in Y:
        out = []
        for letter in w:
            out.append(INV_CHAR_DICT[letter])
        new_Y.append(out)
    return np.asarray(Y)

def training(model, dataloader, learning_rate=0.001, verbose = True):
    loss_fct = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #iterate over batches
    for (batch_id, (X, Y)) in enumerate(dataloader):
        Y = encodeWord(Y)
        optimizer.zero_grad()
        model_out = model(X)
        ctc_input = F.log_softmax(model_out)
        input_lengths = torch.full(size=(len(X),), fill_value=model_out.shape[0], dtype=torch.long)
        #TODO: Check axis
        ctc_target = np.concatenate(Y, axis = 0)
        target_lengths = []
        for w in Y:
            target_lengths.append(len(Y))
        loss = loss_fct(ctc_input, ctc_target, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        if verbose:
            print("Processed Batch {}/{}".format(batch_id, len(dataloader)))
            print("Loss: {}".format(loss))


def data_loader(words_file, data_dir, batch_size):
    #TODO: scale all the inputs to 32x128
    # words_file: absolute path of words.txt
    # data_dir: absolute path of directory that contains the word folders (a01, a02, etc...)
    
    dataset = []

    with open(words_file) as f:
        # line_counter counts the relevant lines to get the desired batch size
        line_counter = 0
        Y = []
        X = []
        for line in f:
            if line_counter < batch_size:
                # skip empty lines and information at the beginning
                if not line.strip() or line[0] == "#":
                    continue
                # construct the image path from the information in the corresponding words.txt lines
                line_split = line.strip().split(' ')
                file_name_split = line_split[0].split('-')
                file_name = '/' + file_name_split[0] + '/' + file_name_split[0] + '-' + file_name_split[1] + '/' + line_split[0] + '.png'
                # load image, convert to greyscale and then to torch tensor
                img = PImage.open(data_dir + file_name).convert('L')
                converter = torchvision.transforms.ToTensor()
                x = converter(img)
                # append the image and the target, obtained from the corresponding words.txt line, to the X,Y lists
                X.append(torch.squeeze(x))
                y = line_split[-1]
                Y.append(y)
                line_counter += 1
            else:
                # add the lists to the dataset and reset the line_counter variable 
                data = (X, Y)
                dataset.append(data)
                Y = []
                X = []
                line_counter = 0
        # if total number of lines is not divisible by the batch_size, the remaining smaller batch must be added at the end
        if len(Y) != 0 and X != 0:
            data = (X,Y)
            dataset.append(data)
        
    
    return dataset
    
    


if __name__=="__main__":
    model = Net()
    n_epochs = 10
    #TODO: Define Dataloader properly (either via pytorch or the github example)
    dataloader = None
    for epoch in range(n_epochs):
        print("Training Epoch "+ str(epoch+1))
        #training(model, dataloader)
        print("Not training due to missing dataloader implementation")

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from PIL import Image as PImage
import torchvision
import math
from random import shuffle
import random

# Here we use '|' as a symbol the CTC-blank
CHAR_LIST = list(" !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz|")
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
    for i in range(len(output)):
        output[i] = "".join(output[i])
    return output

def Best_Path_Decoder(matrix):
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
    # clean the output, i.e. remove multiple letters not seperated by '|' and '|' 
    last_letter = "abc" #invalid label
    current_letter = ""
    output_clean = []
    for i in range(len(output)):
        sub = []
        for j in range(len(output[i])):
            current_letter = output[i][j]
            if output[i][j] != "|" and output[i][j] != last_letter:
                sub.append(output[i][j])
            last_letter = current_letter
        output_clean.append(sub)
    """
    for i in range(len(output)): 
        output[i] = "".join(output[i])
    """
    
    for i in range(len(output_clean)): 
        output_clean[i] = "".join(output_clean[i]).strip()
    #print(output)
    return output_clean

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
        x = x.squeeze(2)
        x = x.permute(0,2,1)
        # pass through LSTM
        x = self.lstm(x)
        x = x[0]
        # transformation for last CNN layer
        x = x.unsqueeze(2)
        x = x.permute(0,3,2,1)
        # pass through last CNN layer
        x = self.cnn(x)
        # transform for CTC_loss calc and text decoding
        x = x.squeeze(2)
        x = x.permute(2,0,1)
        return x

def encodeWord(Y):
    new_Y = []
    for w in Y:
        out = []
        for letter in w:
            out.append(int(INV_CHAR_DICT[letter]))
        new_Y.append(np.asarray(out))
    return new_Y

def training(model, dataloader, learning_rate=0.001, verbose = True):
    loss_fct = nn.CTCLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    #iterate over batches
    for (batch_id, (X, Y)) in enumerate(dataloader):
        Y = encodeWord(Y)
        optimizer.zero_grad()
        model_out = model(X)
        ctc_input = F.log_softmax(model_out)
        input_lengths = torch.full(size=(len(X),), fill_value=model_out.shape[0], dtype=torch.long)
        #print(len(X))
        #TODO: Check axis
        ctc_target = np.concatenate(Y, axis = 0)
        target_lengths = []
        for w in Y:
            target_lengths.append(len(w))
        target_lengths = torch.Tensor(target_lengths).type(torch.int32)
        ctc_target = torch.Tensor(ctc_target).type(torch.int32)
        loss = loss_fct(ctc_input, ctc_target, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        if verbose:
            print("Processed Batch {}/{}".format(batch_id+1, len(dataloader)))
            print("Loss: {}".format(loss))


def data_loader(words_file, data_dir, batch_size, image_size, num_words, train_ratio):
    #TODO: scale all the inputs to 32x128
    # words_file: absolute path of words.txt
    # data_dir: absolute path of directory that contains the word folders (a01, a02, etc...)

    dataset = []

    with open(words_file) as f:
        # line_counter counts the relevant lines to get the desired batch size, counter counts relevant lines, i.e. words
        line_counter = 0
        Y = []
        X = []
        counter = 0
        for line in f:
            if line_counter < batch_size:
                # skip empty lines and information at the beginning
                if not line.strip() or line[0] == "#":
                    continue
                # construct the image path from the information in the corresponding words.txt lines
                line_split = line.strip().split(' ')
                file_name_split = line_split[0].split('-')
                file_name = '/' + file_name_split[0] + '/' + file_name_split[0] + '-' + file_name_split[1] + '/' + line_split[0] + '.png'
                # load image, resize to desired image size, convert to greyscale and then to torch tensor
                try:
                    img = PImage.open(data_dir + file_name).convert('L')
                except:
                    continue
                if counter >= num_words:
                    break
                counter += 1
                (ht, wt) = image_size
                (w, h) = img.size
                fx = w / wt
                fy = h / ht
                f = max(fx, fy)
                new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
                img = img.resize((new_size[0], new_size[1]))
                converter = torchvision.transforms.ToTensor()
                x = converter(img)
                x = torch.squeeze(x)
                x = np.array(x)
                # create target image of size 32x128 and place resized image into it
                target = np.ones([ht, wt]) * 255
                target[0:new_size[1], 0:new_size[0]] = x
                target = torch.tensor(target).float()
                # append the image and the target, obtained from the corresponding words.txt line, to the X,Y lists
                X.append(target)
                y = line_split[-1]
                Y.append(y)
                line_counter += 1
            else:
                # stack the X list to a tensor and add X,Y to the dataset and reset the lists and line_counter variable
                X = torch.stack(X)
                X = torch.unsqueeze(X, 1)
                data = (X, Y)
                dataset.append(data)
                Y = []
                X = []
                line_counter = 0
                if (len(dataset)%100 == 0):
                    print("loaded batch {}/{}".format(len(dataset), math.ceil(115338/batch_size)))
                    print("quitting early for testing")
                    return dataset
        # if total number of lines is not divisible by the batch_size, the remaining smaller batch must be added at the end
        if len(Y) != 0 and len(X) != 0:
            data = (X,Y)
            dataset.append(data)
<<<<<<< HEAD


    return dataset


=======

        # split dataset into train and test set
        random.shuffle(dataset)
        num_of_batches = int(num_words/batch_size)
        num_of_train_batches = int(train_ratio*num_of_batches)
        train_set = dataset[:num_of_train_batches-1]
        test_set = dataset[num_of_train_batches:]

    return train_set, test_set


>>>>>>> b18a420eace48d068efaa605ff0047c02a6dd176


if __name__=="__main__":
    model = Net()
<<<<<<< HEAD
    n_epochs = 100
    words_file = "../dataset/words.txt"
    data_dir = "../dataset/images"
=======
    n_epochs = 50
    words_file = "C:/Users/Musta/Desktop/Uni/MASTER/Veranstaltungen/Advanced Machine Learning/Project/data/words.txt"
    data_dir = "C:/Users/Musta/Desktop/Uni/MASTER/Veranstaltungen/Advanced Machine Learning/Project/data"
>>>>>>> b18a420eace48d068efaa605ff0047c02a6dd176
    batch_size = 50
    image_size = (32, 128)
    num_words = 10000
    train_ratio = 0.9
    train_set, test_set = data_loader(words_file, data_dir, batch_size, image_size, num_words, train_ratio)
    for epoch in range(n_epochs):
        #shuffle data to prevent cyclic effects
        shuffle(dataloader)
        print("Training Epoch "+ str(epoch+1))
<<<<<<< HEAD
        training(model, dataloader, learning_rate=0.1*(1/(epoch+1)))
        #print("Not training due to missing dataloader implementation")
    torch.save(model, "../trained_models/model.chkpt")
=======
        training(model, train_set)
    # testing...
    correct = 0
    counter = 0
    with torch.no_grad():
        for (X, Y) in test_set:
            output = model(X)
            output = np.array(output)
            output = Decoder(output)
            for i in range(len(output)):
                counter += 1
                if output[i] == Y[i]:
                    correct += 1
            #print(output)
            #print(Y)
    print("accuracy:", correct/counter)
>>>>>>> b18a420eace48d068efaa605ff0047c02a6dd176

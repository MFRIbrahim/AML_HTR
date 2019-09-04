import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from PIL import Image as PImage
import torchvision
import math
from random import shuffle
import random
import cv2
from copy import copy
from ConvLSTM_pytorch.convlstm import ConvLSTM

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# Here we use '|' as a symbol the CTC-blank
CHAR_LIST = list("| !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
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
    def __init__(self, lstm_layers=2):
        super(Net, self).__init__()

        self.lstm_layers = lstm_layers
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
        self.lstm = ConvLSTM(input_size=(1,1), input_dim=256, hidden_dim=256, kernel_size=(3,3), num_layers=lstm_layers, batch_first=True)

        #---last CNN layer---
        self.cnn = nn.Conv2d(in_channels=(256) , out_channels=80, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # pass through CNN layers
        for layer in self.cnn_layers:
            x = layer(x)
        # transformation for LSTM
        x = x.permute(0,3,1,2)
        x = x.unsqueeze(4)
        # pass through LSTM
        x, hidden = self.lstm(x)
        x = x[0]
        # transformation for last CNN layer
        x = x.squeeze(3)
        x = x.permute(0,2,3,1)
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

def decodeWord(Y):
    new_Y = []
    for letter in Y:
        new_Y.append(CHAR_DICT[letter])
    return new_Y

def training(model, optimizer, dataloader, learning_rate=0.001, verbose = True):
    loss_fct = nn.CTCLoss().to(device)
    model.train(mode=True)
    mean_loss = 0
    #iterate over batches
    for (batch_id, (X, Y)) in enumerate(dataloader):
        Y = encodeWord(Y)
        X = X.to(device)
        optimizer.zero_grad()
        model_out = model(X)
        ctc_input = F.log_softmax(model_out, dim=-1).to(device)
        input_lengths = torch.full(size=(len(X),), fill_value=model_out.shape[0], dtype=torch.long).to(device)
        #print(len(X))
        #TODO: Check axis
        ctc_target = np.concatenate(Y, axis = 0)
        target_lengths = []
        for w in Y:
            target_lengths.append(len(w))
        target_lengths = torch.Tensor(target_lengths).to(device).type(torch.long)
        ctc_target = torch.Tensor(ctc_target).to(device).type(torch.long)
        if batch_id == 0:
            cpu_input = np.array(copy(ctc_input).detach().cpu())
            out = Best_Path_Decoder(cpu_input)
            for word in out:
        loss = loss_fct(ctc_input, ctc_target, input_lengths, target_lengths)
        mean_loss += loss.item()
        loss.backward()
        optimizer.step()
        if verbose:
            print("Processed Batch {}/{}".format(batch_id+1, len(dataloader)))
            print("Loss: {}".format(loss))
    return mean_loss/len(dataloader)


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
                if x.ndim != 2:
                    continue
                counter += 1
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
        # if total number of lines is not divisible by the batch_size, the remaining smaller batch must be added at the end
        if len(Y) != 0 and len(X) != 0:
            X = torch.stack(X)
            X = torch.unsqueeze(X, 1)
            data = (X,Y)
            dataset.append(data)
        # split dataset into train and test set
        num_of_batches = math.ceil(num_words/batch_size)
        num_of_train_batches = int(train_ratio*num_of_batches)
        train_set = dataset[:num_of_train_batches]
        test_set = dataset[num_of_train_batches:]
        random.shuffle(train_set)
        random.shuffle(test_set)
    return train_set, test_set




if __name__=="__main__":
    model_path = "../trained_models/model_tmp.chkpt"
    epoch = 0
    loss = 0
    weight_decay = 0
    retrain_model = True
    warm_start = False
    model = Net().to(device)
    n_epochs = 20
    words_file = "../dataset/words.txt"
    data_dir = "../dataset/images"
    batch_size = 50
    image_size = (32, 128)
    num_words = 10000
    train_ratio = 0.6
    train_set, test_set = data_loader(words_file, data_dir, batch_size, image_size, num_words, train_ratio)
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if warm_start:
        checkpoint = torch.load("../trained_models/model_optim_tmp.chkpt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    if retrain_model:
        for k in range(n_epochs):
            #shuffle data to prevent cyclic effects
            shuffle(train_set)
            print("Training Epoch "+ str(epoch+1))
            if epoch >= 10:
                lr = 0.001
            if epoch >= 500:
                lr = 0.0001

            loss = training(model, optimizer, train_set, learning_rate=lr, verbose=False)
            print("Loss: {}".format(loss))
            epoch += 1
            if epoch % 10 == 0:
                torch.save({'epoch': epoch, 'loss': loss, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, "../trained_models/model_optim_tmp.chkpt")
                print("saving progress")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        torch.save({'epoch': epoch, 'loss': loss, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, "../trained_models/2DLSTM.chkpt")
    else:
        checkpoint = torch.load("../trained_models/model_optim_tmp.chkpt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    # testing...
    correct = 0
    counter = 0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(test_set):
            X = X.to(device)
            output = F.log_softmax(model(X), dim=-1)
            output = np.array(output.cpu())
            predicted_word = Best_Path_Decoder(output)
            for i in range(len(predicted_word)):
                counter += 1
                if batch < 1:
                    print(predicted_word[i])
                if predicted_word[i] == Y[i]:
                    correct += 1
    print("test accuracy:", correct/counter)
    correct = 0
    counter = 0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(train_set):
            X = X.to(device)
            output = F.log_softmax(model(X), dim=-1)
            output = np.array(output.cpu())
            predicted_word = Best_Path_Decoder(output)
            for i in range(len(predicted_word)):
                if batch < 1:
                    print(predicted_word[i])
                counter += 1
                if predicted_word[i] == Y[i]:
                    correct += 1
    print("train accuracy:", correct/counter)

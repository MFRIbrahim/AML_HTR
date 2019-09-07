import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms

from dataset import get_data_loaders
from transformations import GrayScale, Rescale, ToTensor, word_tensor_to_list
from util import TimeMeasure
from copy import copy
from beam_search import ctcBeamSearch
from data_augmentation import DataAugmenter
from deslant import deslant_image
import ctypes

if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = 'cpu'

torch.manual_seed(0)


# Here we use '|' as a symbol the CTC-blank
CHAR_LIST = list("| !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
CHAR_DICT = {}
for i in range(len(CHAR_LIST)):
    CHAR_DICT[i] = CHAR_LIST[i]
INV_CHAR_DICT = {v: k for k, v in CHAR_DICT.items()}


def decoder(matrix):
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


def best_path_decoder(matrix):
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
    def __init__(self, lstm_layers=2, bidirectional=True, dropout=0.0):
        super(Net, self).__init__()

        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
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
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

        #---last CNN layer---
        self.cnn = nn.Conv2d(in_channels=(bidirectional+1)*(256) , out_channels=80, kernel_size=1, stride=1, padding=0)

        self.init_hidden()

    def init_hidden(self):
        self.hidden = (nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.lstm_layers*(self.bidirectional+1), 50, 256).type(torch.FloatTensor)).to(device), requires_grad=True), nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.lstm_layers*(self.bidirectional+1), 50, 256).type(torch.FloatTensor)), requires_grad=True).to(device))

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


def encode_word(Y):
    new_Y = []
    for w in Y:
        out = []
        for letter in w:
            out.append(int(INV_CHAR_DICT[letter]))
        new_Y.append(np.asarray(out))
    return new_Y


def decode_word(numbers):
    return [CHAR_DICT[num] for num in numbers]


def training(model, optimizer, dataloader, learning_rate=0.001, verbose = True):
    loss_fct = nn.CTCLoss().to(device)
    model.train(mode=True)
    mean_loss = 0
    #iterate over batches
    for (batch_id, (X, Y)) in enumerate(dataloader):
        with TimeMeasure(enter_msg="Running batch", writer=print):
            model.init_hidden()
            X = X.to(device)
            Y = ["".join(decode_word(w)).strip() for w in word_tensor_to_list(Y)]
            print(X.size(), len(Y))
            optimizer.zero_grad()
            model_out = model(X)
            ctc_input = F.log_softmax(model_out).to(device)
            input_lengths = torch.full(size=(len(X),), fill_value=model_out.shape[0], dtype=torch.long).to(device)
            #TODO: Check axis
            ctc_target = np.concatenate(Y, axis = 0)
            target_lengths = []
            for w in Y:
                target_lengths.append(len(w))
            target_lengths = torch.Tensor(target_lengths).long().to(device)
            ctc_target = torch.Tensor(ctc_target).long().to(device)

            if batch_id == 0:
                cpu_input = np.array(copy(ctc_input).detach().cpu())
                out = best_path_decoder(cpu_input)
                for word in out:
                    print(word)

            loss = loss_fct(ctc_input, ctc_target, input_lengths, target_lengths)
            mean_loss += loss.item()
            loss.backward()
            optimizer.step()
            if verbose:
                print("Processed Batch {}/{}".format(batch_id, len(dataloader)))
                print("Loss: {}".format(loss))
    return mean_loss / len(dataloader)


# =====================================================================================================================
# transform = DataAugmenter(p_erase=0, p_jitter=0, p_translate=0, p_perspective=0)
# deslant_image

if __name__ == "__main__":
    width = 32
    height = 128
    max_word_length = 32
    transformation = transforms.Compose([GrayScale(),
                                         Rescale(32, 128, 32),
                                         ToTensor(char_to_int=INV_CHAR_DICT)
                                         ])
    meta_path = "../dataset/words.txt"
    images_path = "../dataset/images"
    relative_train_size = 0.6
    batch_size = 50

    train_loader, test_loader = get_data_loaders(meta_path,
                                                 images_path,
                                                 transformation,
                                                 relative_train_size,
                                                 batch_size)

    model_path = "../trained_models/model_tmp.chkpt"
    epoch = 0
    loss = 0
    weight_decay = 0
    retrain_model = True
    warm_start = False
    model = Net(dropout=0.2).to(device)
    n_epochs = 10
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
            print("Training Epoch " + str(epoch + 1))
            if epoch >= 10:
                lr = 0.001
            if epoch >= 500:
                lr = 0.00005

            loss = training(model, optimizer, train_loader, learning_rate=lr, verbose=False)
            print("Loss: {}".format(loss))
            epoch += 1
            if epoch % 10 == 0:
                torch.save({'epoch': epoch, 'loss': loss, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, "../trained_models/model_optim_tmp.chkpt")
                print("saving progress")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        torch.save({'epoch': epoch, 'loss': loss, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, "../trained_models/ADAM_2LSTM.chkpt")
    else:
        checkpoint = torch.load("../trained_models/ADAM_2LSTM.chkpt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    # testing...
    correct = 0
    counter = 0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(test_loader):
            model.init_hidden()
            X = X.to(device)
            output = F.softmax(model(X), dim=-1)
            output = np.array(output.cpu())
            #predicted_word = best_path_decoder(output)
            predicted_word = ctcBeamSearch(output, "".join(CHAR_LIST), None, beamWidth=4)
            for i in range(len(predicted_word)):
                counter += 1
                if batch < 1:
                    print(predicted_word[i])
                    pass
                if predicted_word[i] == Y[i]:
                    correct += 1
    print("test accuracy:", correct / counter)
    correct = 0
    counter = 0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(train_loader):
            model.init_hidden()
            X = X.to(device)
            output = F.softmax(model(X), dim=-1)
            output = np.array(output.cpu())
            #predicted_word = best_path_decoder(output)
            predicted_word = ctcBeamSearch(output, "".join(CHAR_LIST), None, beamWidth=4)
            for i in range(len(predicted_word)):
                if batch < 1:
                    print(predicted_word[i])
                    pass
                counter += 1
                if predicted_word[i] == Y[i]:
                    correct += 1
    print("train accuracy:", correct / counter)




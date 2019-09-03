import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from PIL import Image as PImage
import cv2
import torchvision
import math
import os

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


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
    loss_fct = nn.CTCLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #iterate over batches
    for (batch_id, (X, Y)) in enumerate(dataloader):
        Y = encodeWord(Y)
        X = X.to(device)
        optimizer.zero_grad()
        model_out = model(X)
        ctc_input = F.log_softmax(model_out).to(device)
        input_lengths = torch.full(size=(len(X),), fill_value=model_out.shape[0], dtype=torch.long).to(device)
        #TODO: Check axis
        ctc_target = np.concatenate(Y, axis = 0)
        target_lengths = []
        for w in Y:
            target_lengths.append(len(w))
        target_lengths = torch.Tensor(target_lengths).to(device).type(torch.int32)
        ctc_target = torch.Tensor(ctc_target).to(device).type(torch.int32)
        loss = loss_fct(ctc_input, ctc_target, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        if verbose:
            print("Processed Batch {}/{}".format(batch_id, len(dataloader)))
            print("Loss: {}".format(loss))


def data_loader(words_file, data_dir, batch_size, image_size):
    # words_file: absolute path of words.txt
    # data_dir: absolute path of directory that contains the word folders (a01, a02, etc...)

    dataset = []

    with open(words_file) as f:
        # line_counter counts the relevant lines to get the desired batch size
        line_counter = 0
        Y = []
        X = torch.empty((batch_size, 1, image_size[0], image_size[1]))
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
                try:
                    img = cv2.imread(data_dir + file_name)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                except:
                    #skip broken images
                    continue
                img = cv2.resize(gray, (image_size[1], image_size[0]))
                converter = torchvision.transforms.ToTensor()
                x = converter(img)
                # append the image and the target, obtained from the corresponding words.txt line, to the X,Y lists
                X[line_counter, 0] = torch.squeeze(x)
                y = line_split[-1]
                Y.append(y)
                line_counter += 1
            else:
                # add the lists to the dataset and reset the lists and line_counter variable
                data = (X, Y)
                dataset.append(data)
                Y = []
                X = torch.empty((batch_size, 1, image_size[0], image_size[1]))
                line_counter = 0
                if (len(dataset)%10 == 0):
                    print("loaded batch {}/{}".format(len(dataset), math.ceil(115338/batch_size)))
                    print("quitting early for testing")
                    return dataset
        # if total number of lines is not divisible by the batch_size, the remaining smaller batch must be added at the end
        if len(Y) != 0 and X != 0:
            short_X = torch.empty((len(Y), 1, image_size[0], image_size[1]))
            short_X[:len(Y),:,:,:] = X[:len(Y),:,:,:]
            data = (short_X,Y)
            dataset.append(data)


    return dataset


def is_file(path):
    return os.path.exists(path) and os.path.isfile(path)


class WordsDataSet(Dataset):
    def __init__(self, meta_file, root_dir, transform=None):
        self.__meta_file = meta_file
        self.__words = list()
        self.__root_dir = root_dir
        self.__transform = transform

        self.__process_meta_file()
        self.__availability_check()

    def __process_meta_file(self):
        with open(self.__meta_file, 'r') as fp:
            for line in fp:
                self.__process_meta_line(line)

    def __process_meta_line(self, line):
        if not line.startswith("#"):
            self.__words.append(WordsMetaData.parse(line))

    def __availability_check(self):
        to_delete = []
        for idx, word_meta in enumerate(self.__words):
            wid = word_meta.word_id
            folder, subfolder, counter, sub_counter = wid.split("-")
            path = os.path.join(self.__root_dir, folder, folder + "-" + subfolder, wid + ".png")
            if not is_file(path):
                print("File not found:", path)
                to_delete.append(idx)

        for idx in sorted(to_delete, key=lambda x: -x):
            del self.__words[idx]

    def __len__(self):
        return len(self.__words)

    def __getitem__(self, item):
        pass


class BoundingBox(object):
    def __init__(self, x, y, w, h):
        self.__x = x
        self.__y = y
        self.__w = w
        self.__h = h

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def pos(self):
        return self.x, self.y

    @property
    def w(self):
        return self.__w

    @property
    def width(self):
        return self.w

    @property
    def h(self):
        return self.__h

    @property
    def height(self):
        return self.h


class WordsMetaData(object):
    def __init__(self, wid, segmentation_state, gray_level, bounding_box, pos_tag, transcription):
        """

        :param wid: word id
        :param segmentation_state: 'ok' - word was correctly, 'err' - segmentation of word can be bad
        :param gray_level: graylevel to binarize the line containing this word
        :param bounding_box: bounding box around this word
        :param pos_tag: the grammatical tag for this word
        :param transcription: the transcription for this word
        """
        self.__wid = wid
        self.__segmentation_state = segmentation_state
        self.__gray_level = gray_level
        self.__bounding_box = bounding_box
        self.__pos_tag = pos_tag
        self.__transcription = transcription

    @property
    def word_id(self):
        return self.__wid

    @property
    def segmentation_state(self):
        return self.__segmentation_state

    @property
    def gray_level(self):
        return self.gray_level

    @property
    def bounding_box(self):
        return self.__bounding_box

    @property
    def pos_tag(self):
        return self.__pos_tag

    @property
    def transcription(self):
        return self.__transcription

    @staticmethod
    def parse(line):
        line = line.strip()
        parts = line.split(" ")
        wid = parts[0]
        state = parts[1]
        gray_level = parts[2]
        pos_tag = parts[7]
        transcription = parts[8]
        box = BoundingBox(x=parts[3], y=parts[4], w=parts[5], h=parts[6])
        return WordsMetaData(wid, state, gray_level, box, pos_tag, transcription)


if __name__ == "__main__":
    #model = Net().to(device)
    #n_epochs = 10
    #TODO: Define Dataloader properly (either via pytorch or the github example)
    #dataloader = data_loader("../dataset/words.txt", "../dataset/images", 50, (32, 128))
    #print(len(dataloader))
    #for epoch in range(n_epochs):
    #    print("Training Epoch "+ str(epoch+1))
    #    training(model, dataloader)

    data = WordsDataSet("../dataset/words.txt", "../dataset/images")
    print("Length:", len(data))

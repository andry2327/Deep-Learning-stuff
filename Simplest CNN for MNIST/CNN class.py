import torch
import torch.nn as nn
import torch.nn.functional as F


'''

ARCHITECTURE:

INPUT tensor (batch size, 1, 28, 28) / 1: grey scale channel


considering a single image of shape (1, 28, 28):
1. x = conv1(INPUT): 32 3 x 3 filter on each image -> x.shape = (32, )


'''

class CNN(nn.Module):

    # initialize NN architetures and its layers
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(in_features=64*22*22, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*22*22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


'''

ARCHITECTURE:

INPUT tensor (batch size, 1, 28, 28) / 1: grey scale channel


considering a single image x of shape (1, 28, 28):

1. x = conv1(INPUT): 32 3 x 3 filter on each image 
    -> x.shape = (32, 26, 26) 
    / output.shape = [(input_size - kernel_size + 2*padding) / stride] + 1 = [(28 - 3 - 2*0) / 1] + 1 = 26

    1.1 x = ReLU(x), x.shape unchanged

2. x = conv2(x): 64 3 x 3 filters (filters.shape = (32, 3, 3)) applied on feature map with shape (32, 26, 26)
    -> x.shape = (64, 24, 24) / [(26 - 3 - 2*0) / 1] + 1 = 24

    2.1 x = ReLU(x), x.shape unchanged

3. x = x.view(-1, 64*22*22): used to flatten the 3D tensor
    -> x.shape: (64, 24, 24) -> (64*24*24)
    / -1: it is for the number of batches, so for each batch -> (64*24*24) vecotr passed to FC layer

4. x = fc1(x): the vector (64*24*24) is used as input for a fully connected layer with 128 output neurons
    -> x.shape: (64*24*24) -> (128)

    4.1 x = ReLU(x), x.shape unchanged

5. x = fc2(x): hidden layer of (128) is used to produce the 10 final classes of each digits
    -> x.shape: (128) -> (10)

    
-------------------------------------------


ARCHITECTURE SUMMARY:

INPUT (shape = (1, 28, 28)) -> conv1 (-> ReLU) -> conv2 (-> ReLU) -> Flattened tensor: x.shape (64, 24, 24) -> (64*24*24) -> fc1 (-> ReLU) -> fc2 -> OUTPUT CLASSES


'''


class CNN(nn.Module):

    # initialize NN architetures and its layers
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(in_features=64*24*24, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*24*24)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

""" Model classes defined here! """

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from skimage.transform import rotate


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as member variables.
        """
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Compute the forward pass of our model, which outputs logits.
        """
        # TODO: Implement this!
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class SimpleConvNN(torch.nn.Module):
    def __init__(self, n1_chan, n1_kern, n2_kern):
        super(SimpleConvNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, n1_chan, kernel_size=n1_kern)
        self.conv2 = torch.nn.Conv2d(n1_chan, 10, kernel_size=n2_kern)
        self.relu = torch.nn.ReLU()
        self.size1 = int(28 - n1_kern + 1)
        self.size2 = int((self.size1 - n2_kern) / 2 + 1)
        self.maxPool2d = torch.nn.MaxPool2d(self.size2)

    def forward(self, x):
        # TODO: Implement this!
        batch = x.shape[0]
        x = x.view(batch, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxPool2d(x)
        x = x.view(batch, -1)
        return x


class BestNN(torch.nn.Module):
    # TODO: You can change the parameters to the init method if you need to
    # take hyperparameters from the command line args!
    def __init__(self, n1_chan, n1_kern, n2_kern, linear_size, dropout):
        super(BestNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, n1_chan, n1_kern)
        self.norm1 = torch.nn.BatchNorm2d(n1_chan)
        self.conv2 = torch.nn.Conv2d(n1_chan, 40, n2_kern)
        self.norm2 = torch.nn.BatchNorm2d(40)
        self.size = int(((28 - n1_kern + 1) / 2 - n2_kern + 1) / 2)
        self.fc1 = torch.nn.Linear(self.size * self.size * 40, linear_size)
        self.fc2 = torch.nn.Linear(linear_size, 10)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch, 1, 28, 28)
        for img in x:
            angle = np.random.uniform(low=-20.0, high=20.0)
            flag = np.random.randint(0, 5)
            if flag == 0:
                img = rotate(img, angle)
            if flag == 1:
                img = ndimage.gaussian_filter(img, sigma=2)
            if flag == 2:
                img = ndimage.interpolation.shift(img, (5, 0, 0), order=0, mode='nearest')
            if flag == 3:
                img = ndimage.interpolation.shift(img, (-5, 0, 0), order=0, mode='nearest')
            '''if flag == 4:
                img = rotate(img, angle)
                img = ndimage.gaussian_filter(img, sigma=3)
                img = ndimage.interpolation.shift(img, (0, 2, 0), order=0, mode='nearest')
            else:
                img = rotate(img, angle)
                img = ndimage.gaussian_filter(img, sigma=3)
                img = ndimage.interpolation.shift(img, (1, 0, 0), order=0, mode='nearest')'''

        x = F.relu(self.norm1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.size * self.size * 40)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

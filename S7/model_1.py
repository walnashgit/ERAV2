# model
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, 3, padding = 1) # RF: 3
    self.conv2 = nn.Conv2d(16, 16, 3, padding = 1) # RF: 5
    self.pool1 = nn.MaxPool2d(2, 2) # RF: 6

    self.conv3 = nn.Conv2d(16, 10, 3, padding = 1) # RF: 10
    self.conv4 = nn.Conv2d(10, 10, 3, padding = 1) # RF: 14
    self.pool2 = nn.MaxPool2d(2, 2) # RF: 16

    self.conv5 = nn.Conv2d(10, 16, 3) # RF: 20
    self.conv6 = nn.Conv2d(16, 16, 3) # RF: 24
    self.conv7 = nn.Conv2d(16, 10, 3) # RF: 28

  def forward(self, x):
    x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
    x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
    x = F.relu(self.conv6(F.relu(self.conv5(x))))
    x = self.conv7(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)
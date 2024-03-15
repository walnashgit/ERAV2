# model
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()

    # Input Block
    self.convblock1 = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding = 1, bias = False), # RF: 3, OP: 28
        nn.BatchNorm2d(16),
        nn.ReLU()
    ) 

    # CONV block 1
    self.convblock2 = nn.Sequential(
        nn.Conv2d(16, 16, 3, padding = 1, bias = False), # RF: 5, OP: 28
        nn.BatchNorm2d(16),
        nn.ReLU()
    ) 

    # Transition block 1
    self.pool1 = nn.MaxPool2d(2, 2) # RF: 6, OP: 14
    self.convblock3 = nn.Sequential(
        nn.Conv2d(16, 10, 1, padding = 0, bias = False), # RF: 6, OP: 14
        nn.BatchNorm2d(10),
        nn.ReLU()
    ) 

    # CONV block 2
    self.convblock4 = nn.Sequential(
        nn.Conv2d(10, 10, 3, padding = 1, bias = False), # RF: 10, OP: 14
        nn.BatchNorm2d(10),
        nn.ReLU()
    ) 

    # Transition block 2
    self.pool2 = nn.MaxPool2d(2, 2) # RF: 12, OP: 7
    self.convblock5 = nn.Sequential(
        nn.Conv2d(10, 16, 1, padding = 0, bias = False), # RF: 12, OP: 7
        nn.BatchNorm2d(16),
        nn.ReLU()
    ) 

    # OUTPUT Block
    self.convblock6 = nn.Sequential(
        nn.Conv2d(16, 16, 3, padding = 1, bias = False), # RF: 20, OP: 7
        nn.BatchNorm2d(16),
        nn.ReLU()
    ) 
    self.convblock7 = nn.Sequential(
        nn.Conv2d(16, 10, 3, padding = 0, bias = False), # RF: 28, OP: 5
    ) 

    # Dropout 5%
    self.drop = nn.Dropout2d(0.05)

    self.gap1 = nn.AdaptiveAvgPool2d(1) #OP: 1

    

  def forward(self, x):
    x = self.convblock1(x)
    x = self.drop(x)

    x = self.convblock2(x)
    x = self.drop(x)

    #Transition 1
    x = self.pool1(x)
    x = self.convblock3(x)
    x = self.drop(x)

    x = self.convblock4(x)
    x = self.drop(x)

    # Transition 2
    x = self.pool2(x)
    x = self.convblock5(x)
    x = self.drop(x)

    x = self.convblock6(x)
    x = self.drop(x)

    x = self.convblock7(x)

    x = self.gap1(x)

    x = x.view(-1, 10)

    return F.log_softmax(x, dim=-1)
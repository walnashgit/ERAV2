import torch.nn as nn
import torch.nn.functional as F

# S6 model. accuracy 99.41%

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias = False) # param =  #input -? OUtput? RF
        self.batchN1 = nn.BatchNorm2d(32)


        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias = False) # param =
        self.batchN2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)


        self.conv3 = nn.Conv2d(32, 16, 3, padding=1, bias = False) # param =
        self.batchN3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(16, 16, 3, bias = False) #

        self.conv5 = nn.Conv2d(16, 16, 3, bias = False)

        self.gap1 = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(16, 10)

        self.drop = nn.Dropout2d(0.05)


    def forward(self, x):
        #L 1
        #print("x shape1", x.shape)
        x = self.batchN1(F.relu(self.conv1(x)))
        x = self.drop(x)

        # L 2
        #print("x shape2", x.shape)
        x = self.pool2(self.batchN2(F.relu(self.conv2(x))))
        x = self.drop(x)

        # L 3
        #print("x shape3", x.shape)
        x = F.relu(self.conv3(x))
        #print("x shape4", x.shape)
        self.batchN3(x)
        #print("x shape5", x.shape)
        x = self.pool3(x)

        # L 4
        #print("x shape6", x.shape)
        x = F.relu(self.conv4(x))

        # L 5
        #print("x shape6", x.shape)
        x = F.relu(self.conv5(x))

        # L 6
        #print("x shape1", x.shape)
        x = self.gap1(x)

        #print("x shape2", x.shape)
        x = x.view(-1, 16)

        #print("x shape9", x.shape)
        x = self.fc1(x)

        #print("\nx shape4", x.shape)
        return F.log_softmax(x, dim = 1)


# S7 Model - 1


class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, 3, padding = 1) # RF: 3, OP: 28
    self.conv2 = nn.Conv2d(16, 16, 3, padding = 1) # RF: 5, OP: 28
    self.pool1 = nn.MaxPool2d(2, 2) # RF: 6, OP: 14

    self.conv3 = nn.Conv2d(16, 10, 3, padding = 1) # RF: 10, OP: 14
    self.conv4 = nn.Conv2d(10, 10, 3, padding = 1) # RF: 14, OP: 14
    self.pool2 = nn.MaxPool2d(2, 2) # RF: 16, OP: 7

    self.conv5 = nn.Conv2d(10, 16, 3) # RF: 24, OP: 5
    self.conv6 = nn.Conv2d(16, 16, 3) # RF: 32, OP: 3
    self.conv7 = nn.Conv2d(16, 10, 3) # RF: 40, OP: 1

  def forward(self, x):
    x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
    x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
    x = F.relu(self.conv6(F.relu(self.conv5(x))))
    x = self.conv7(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)


# S7 Model - 2


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),  # RF: 3, OP: 28
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # CONV block 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),  # RF: 5, OP: 28
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Transition block 1
        self.pool1 = nn.MaxPool2d(2, 2)  # RF: 6, OP: 14
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False),  # RF: 6, OP: 14
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # CONV block 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),  # RF: 10, OP: 14
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # Transition block 2
        self.pool2 = nn.MaxPool2d(2, 2)  # RF: 12, OP: 7
        self.convblock5 = nn.Sequential(
            nn.Conv2d(10, 16, 1, padding=0, bias=False),  # RF: 12, OP: 7
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # OUTPUT Block
        self.convblock6 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),  # RF: 20, OP: 7
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(16, 10, 3, padding=0, bias=False),  # RF: 28, OP: 5
        )

        # Dropout 5%
        self.drop = nn.Dropout2d(0.05)

        self.gap1 = nn.AdaptiveAvgPool2d(1)  # OP: 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.drop(x)

        x = self.convblock2(x)
        x = self.drop(x)

        # Transition 1
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


# S7 Model - 3


dropout_value = 0.05

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),  # RF: 3, OP: 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # CONV block 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),  # RF: 5, OP: 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # Transition block 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False),  # RF: 5, OP: 28
            # nn.ReLU(),
            # nn.BatchNorm2d(10)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # RF: 6, OP: 14

        # CONV block 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),  # RF: 10, OP: 14
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        self.pool2 = nn.MaxPool2d(2, 2)  # RF: 12, OP: 7

        self.convblock5 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=0, bias=False),  # RF: 20, OP: 5
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # Transition block 2
        # self.convblock6 = nn.Sequential(
        #     nn.Conv2d(16, 10, 1, padding=0, bias=False),  # RF: 10, OP: 14
        #     # nn.ReLU(),
        #     # nn.BatchNorm2d(10)
        # )
        # self.pool2 = nn.MaxPool2d(2, 2)  # RF: 16, OP: 7

        # OUTPUT Block
        self.convblock6 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0, bias=False),  # RF: 28, OP: 3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # self.convblock7 = nn.Sequential(
        #     nn.Conv2d(10, 10, 3, padding=0, bias=False),  # RF: 32, OP: 3
        # )

        # Dropout 5%
        # self.drop = nn.Dropout2d(0.05)

        self.gap1 = nn.AdaptiveAvgPool2d(1)  # RF: 28, OP: 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding = 0, bias = False), # RF: 28, OP: 1
        )

        #self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.convblock1(x)

        x = self.convblock2(x)

        # Transition 1
        x = self.convblock3(x)
        x = self.pool1(x)

        x = self.convblock4(x)

        x = self.pool2(x)

        x = self.convblock5(x)

        x = self.convblock6(x)

        #x = self.convblock7(x)

        x = self.gap1(x)

        x = self.convblock8(x)

        x = x.view(-1, 10)
        #x = self.fc1(x)

        return F.log_softmax(x, dim=1)


# S8 Model 1 - Batch Normalization


dropout_value = 0.05

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),  # RF: 3, OP: 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF: 5, OP: 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # c3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0, bias=False),  # RF: 5, OP: 32
        )
        # P1
        self.pool1 = nn.MaxPool2d(2, 2)  # RF: 6, OP: 16

        # C4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=1, bias=False),  # RF: 10, OP: 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),  # RF: 14, OP: 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF: 18, OP: 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # c7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0, bias=False),  # RF: 18, OP: 16
        )
        # P2
        self.pool2 = nn.MaxPool2d(2, 2)  # RF:20, OP: 8

        # C8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1, bias=False),  # RF: 28, OP: 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C9
        self.convblock9 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF: 36, OP: 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C10
        self.convblock10 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF:44, OP: 8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # GAP
        self.gap1 = nn.AdaptiveAvgPool2d(1)  # OP: [X, 32, 1, 1] ; X = Batch size

        # c11
        self.convblock11 = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0, bias=False),  # OP: [X, 10, 1, 1]
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.convblock3(x)
        x = self.pool1(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)

        x = self.convblock7(x)
        x = self.pool2(x)

        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)

        x = self.gap1(x)

        x = self.convblock11(x)

        x = x.view(-1, 10)  # OP: [128, 10]

        x = F.log_softmax(x, dim=-1)

        return x


# S8 Model - Group Normalization


dropout_value = 0.05

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),  # RF: 3, OP: 32
            nn.GroupNorm(4, 32),  # grp, ch
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF: 5, OP: 32
            nn.GroupNorm(4, 32),  # grp, ch
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # c3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0, bias=False),  # RF: 5, OP: 32
        )
        # P1
        self.pool1 = nn.MaxPool2d(2, 2)  # RF: 6, OP: 16

        # C4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=1, bias=False),  # RF: 10, OP: 16
            nn.GroupNorm(4, 16),  # grp, ch
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),  # RF: 14, OP: 16
            nn.GroupNorm(8, 32),  # grp, ch
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF: 18, OP: 16
            nn.GroupNorm(8, 32),  # grp, ch
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # c7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0, bias=False),  # RF: 18, OP: 16
        )
        # P2
        self.pool2 = nn.MaxPool2d(2, 2)  # RF:20, OP: 8

        # C8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1, bias=False),  # RF: 28, OP: 8
            nn.GroupNorm(4, 32),  # grp, ch
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C9
        self.convblock9 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF: 36, OP: 8
            nn.GroupNorm(4, 32),  # grp, ch
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C10
        self.convblock10 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF:44, OP: 8
            nn.GroupNorm(4, 32),  # grp, ch
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # GAP
        self.gap1 = nn.AdaptiveAvgPool2d(1)  # OP: [X, 32, 1, 1] ; X = Batch size

        # c11
        self.convblock11 = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0, bias=False),  # OP: [X, 10, 1, 1]
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.convblock3(x)
        x = self.pool1(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)

        x = self.convblock7(x)
        x = self.pool2(x)

        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)

        x = self.gap1(x)

        x = self.convblock11(x)

        x = x.view(-1, 10)  # OP: [128, 10]

        x = F.log_softmax(x, dim=-1)

        return x


# S8 Model - 3 - Layer Normalization


dropout_value = 0.1

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # C1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1, bias=False),  # RF: 3, OP: 32
            nn.LayerNorm([12, 32, 32]),  # Ch, Ht, Width
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 5, OP: 32
            nn.LayerNorm([12, 32, 32]),  # Ch, Ht, Width
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # c3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(12, 10, 1, padding=0, bias=False),  # RF: 5, OP: 32
        )
        # P1
        self.pool1 = nn.MaxPool2d(2, 2)  # RF: 6, OP: 16

        # C4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(10, 12, 3, padding=1, bias=False),  # RF: 10, OP: 16
            nn.LayerNorm([12, 16, 16]),  # Ch, Ht, Width
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 14, OP: 16
            nn.LayerNorm([12, 16, 16]),  # Ch, Ht, Width
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 18, OP: 16
            nn.LayerNorm([12, 16, 16]),  # Ch, Ht, Width
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # c7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(12, 10, 1, padding=0, bias=False),  # RF: 18, OP: 16
        )
        # P2
        self.pool2 = nn.MaxPool2d(2, 2)  # RF:20, OP: 8

        # C8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(10, 12, 3, padding=1, bias=False),  # RF: 28, OP: 8
            nn.LayerNorm([12, 8, 8]),  # Ch, Ht, Width
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C9
        self.convblock9 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF: 36, OP: 8
            nn.LayerNorm([12, 8, 8]),  # Ch, Ht, Width
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C10
        self.convblock10 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),  # RF:44, OP: 8
            nn.LayerNorm([12, 8, 8]),  # Ch, Ht, Width
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # GAP
        self.gap1 = nn.AdaptiveAvgPool2d(1)  # OP: [X, 32, 1, 1] ; X = Batch size

        # c11
        self.convblock11 = nn.Sequential(
            nn.Conv2d(12, 10, 1, padding=0, bias=False),  # OP: [X, 10, 1, 1]
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.convblock3(x)
        x = self.pool1(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)

        x = self.convblock7(x)
        x = self.pool2(x)

        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)

        x = self.gap1(x)

        x = self.convblock11(x)

        x = x.view(-1, 10)  # OP: [128, 10]

        x = F.log_softmax(x, dim=-1)

        return x

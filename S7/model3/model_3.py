# model best: 99.32 17th epoch
import torch.nn as nn
import torch.nn.functional as F

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
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.05

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # C1 - last conv uses stride of 2
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),  # RF: 3, OP: 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF: 5, OP: 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value),

            nn.Conv2d(32, 32, 3, stride=2, bias=False),  # RF: 7, OP: 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C2 normal convolution
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF: 11, OP: 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF: 15, OP: 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),  # RF: 19, OP: 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )

        # C3 Depthwise separable convolution and dilated conv; groups = in_channel_size
        # for depthwise separable convolution, after conv with group = in_channel_size,
        # it must follow a conv with 1x1 kernel
        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False, dilation=2),  # RF: 27, OP: 14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value),
            nn.Conv2d(32, 64, kernel_size=1, bias=False),

            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False, dilation=2),  # RF: 35, OP: 12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_value),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),

            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False, dilation=2),  # RF: 43, OP: 10
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_value),
            nn.Conv2d(128, 128, kernel_size=1, bias=False)
        )

        # C4 Depthwise separable convolution; groups = in_channel_size
        self.convblock4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, groups=128, bias=False),  # RF: 47, OP: 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_value),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),

            nn.Conv2d(128, 128, 3, groups=128, bias=False),   # RF: 51, OP: 6
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_value),
            nn.Conv2d(128, 256, kernel_size=1, bias=False),

            nn.Conv2d(256, 256, 3, groups=256, bias=False),  # RF: 55, OP: 4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_value),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
        )

        # GAP
        self.gap1 = nn.AdaptiveAvgPool2d(1)

        self.convblockTarget = nn.Sequential(
            nn.Conv2d(256, 10, 1, padding=0, bias=False),  # OP: [X, 10, 1, 1]
        )

    def forward(self, x):
        x = self.convblock1(x)
        # print("X shape after C1: ", x.shape)
        x = self.convblock2(x)
        # print("X shape after C2: ", x.shape)
        x = self.convblock3(x)
        # print("X shape after C3: ", x.shape)
        x = self.convblock4(x)

        x = self.gap1(x)

        x = self.convblockTarget(x)

        x = x.view(-1, 10)  # OP: [128, 10]

        x = F.log_softmax(x, dim=-1)

        return x

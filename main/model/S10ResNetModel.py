import torch.nn as nn

dropout_value = 0.05


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.LayerPrep = BasicBlock(3, 64, mx_pool=False, padding=1)

        self.Layer1Basic = BasicBlock(64, 128, mx_pool=True, padding=1)
        self.Layer1ResNet = ResnetBlock(128)

        self.Layer2 = BasicBlock(128, 256, mx_pool=True, padding=1)

        self.Layer3Basic = BasicBlock(256, 512, mx_pool=True, padding=1)
        self.Layer3ResNet = ResnetBlock(512)

        self.mx_pool = nn.MaxPool2d(4, 4)

        self.fc = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = self.LayerPrep(x)

        x = self.Layer1Basic(x)
        y = self.Layer1ResNet(x)
        x = x + y

        x = self.Layer2(x)

        x = self.Layer3Basic(x)
        y = self.Layer3ResNet(x)
        x = x + y

        x = self.mx_pool(x)

        # print("X shape bf x.view: ", x.shape)
        x = x.view(-1, 512)
        # print("X shape bf fc: ", x.shape)
        x = self.fc(x)

        # return F.log_softmax(x, dim=1)
        return x  # F.softmax(x, dim=1)


class ResnetBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResnetBlock, self).__init__()

        self.resnet = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, groups=in_channel, padding=1, bias=False),  # RF: , OP:
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.Dropout2d(dropout_value),

            nn.Conv2d(in_channel, in_channel, 3, groups=in_channel, padding=1, bias=False),  # RF: , OP:
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
            nn.Dropout2d(dropout_value)
        )

    def forward(self, x):

        return self.resnet(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, padding, mx_pool=False):
        super(BasicBlock, self).__init__()

        layers = [
            # nn.Conv2d(in_channel, in_channel, 3, groups=in_channel, padding=padding, bias=False),
            nn.Conv2d(in_channel, out_channel, 3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            # nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.Dropout2d(dropout_value)
        ]
        if mx_pool:
            layers.insert(1, nn.MaxPool2d(2, 2))

        self.basic = nn.Sequential(*layers)

    def forward(self, x):
        return self.basic(x)



from typing import overload
import torch.nn as nn
import torch.nn.functional as F
from .resnet import BasicBlock, ResNet, conv1x1, conv3x3

import ipdb

class DoubleNetIndep(ResNet):
    def __init__(self, block, num_layers, compress_rate, num_classes=10):
        super(DoubleNetIndep, self).__init__(block, num_layers, compress_rate) 
        self.br1 = nn.Sequential(
            nn.Conv2d(self.overall_channel[0], self.overall_channel[9], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.overall_channel[9]),
            nn.ReLU(inplace=True),
        )
        self.br2 = nn.Sequential(
            nn.Conv2d(self.overall_channel[9], self.overall_channel[18], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.overall_channel[18]),
            nn.ReLU(inplace=True),
        )
        self.br3 = nn.Sequential(
            nn.Conv2d(self.overall_channel[18], self.overall_channel[27], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.overall_channel[27]),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.br1(x)
        for i, block in enumerate(self.layer1):
            x = block(x)
        x = x + x1

        x1 = self.br2(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        x = x + x1

        x1 = self.br3(x)
        for i, block in enumerate(self.layer3):
            x = block(x)
        x = x + x1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.num_layer == 56:
            x = self.fc(x)
        else:
            x = self.linear(x)

        return x

class DoubleNetIndepBottleneck(ResNet):
    def __init__(self, block, num_layers, compress_rate, num_classes=10):
        super(DoubleNetIndepBottleneck, self).__init__(block, num_layers, compress_rate) 
        self.br1 = nn.Sequential(
            conv1x1(self.overall_channel[0], 16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            conv3x3(16, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            conv1x1(16, self.overall_channel[9]),
            nn.BatchNorm2d(self.overall_channel[9]),
            nn.ReLU(inplace=True),
        )
        self.br2 = nn.Sequential(
            conv1x1(self.overall_channel[9], 16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            conv3x3(16, 32, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv1x1(32, self.overall_channel[18]),
            nn.BatchNorm2d(self.overall_channel[18]),
            nn.ReLU(inplace=True),
        )
        self.br3 = nn.Sequential(
            conv1x1(self.overall_channel[18], 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv3x3(32, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv1x1(64, self.overall_channel[27]),
            nn.BatchNorm2d(self.overall_channel[27]),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.br1(x)
        for i, block in enumerate(self.layer1):
            x = block(x)
        x = x + x1

        x1 = self.br2(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        x = x + x1

        x1 = self.br3(x)
        for i, block in enumerate(self.layer3):
            x = block(x)
        x = x + x1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.num_layer == 56:
            x = self.fc(x)
        else:
            x = self.linear(x)

        return x


def doublenet_56_indep(compress_rate):
    return DoubleNetIndep(BasicBlock, 56, compress_rate=compress_rate)

def doublenet_56_indep_bottleneck(compress_rate):
    return DoubleNetIndepBottleneck(BasicBlock, 56, compress_rate=compress_rate)

def doublenet_110(compress_rate):
    return DoubleNetIndep(BasicBlock, 110, compress_rate=compress_rate)
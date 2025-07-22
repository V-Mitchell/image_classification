import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dict import remove_key


class SimpleModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ClassifierModel, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(6, 16, 5)
        self.fc0 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv0(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 5 * 5)  # 5x5 feature map
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ClassifierModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ClassifierModel, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, 6, 5)
        self.conv1 = nn.Conv2d(6, 12, 5)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc0 = nn.Linear(24 * 20 * 20, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = x.view(-1, 24 * 20 * 20)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, first_block=False):
        super(Bottleneck, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = nn.Sequential()
        if first_block:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * self.expansion,
                          kernel_size=1,
                          stride=stride,
                          padding=0), nn.BatchNorm2d(out_channels * self.expansion))

    def forward(self, x):
        y = self.relu(self.bn0(self.conv0(x)))
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))
        y += self.downsample(x)
        return self.relu(y)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, first_block=False):
        super(BasicBlock, self).__init__()

        self.conv0 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.stride = stride
        self.downsample = nn.Sequential()
        if first_block and stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = self.relu(self.bn0(self.conv0(x)))
        y = self.bn1(self.conv1(y))
        y += self.downsample(x)
        return self.relu(y)


class ResNet(nn.Module):
    in_channels = 64

    def __init__(self,
                 ResBlock,
                 blocks_list,
                 out_channels_list=[64, 128, 256, 512],
                 in_channels=3,
                 num_classes=10):
        super(ResNet, self).__init__()

        self.conv0 = nn.Conv2d(in_channels,
                               self.in_channels,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self.create_layer(ResBlock,
                                        blocks_list[0],
                                        self.in_channels,
                                        out_channels_list[0],
                                        stride=1)
        self.layer1 = self.create_layer(ResBlock,
                                        blocks_list[1],
                                        out_channels_list[0] * ResBlock.expansion,
                                        out_channels_list[1],
                                        stride=2)
        self.layer2 = self.create_layer(ResBlock,
                                        blocks_list[2],
                                        out_channels_list[1] * ResBlock.expansion,
                                        out_channels_list[2],
                                        stride=2)
        self.layer3 = self.create_layer(ResBlock,
                                        blocks_list[3],
                                        out_channels_list[2] * ResBlock.expansion,
                                        out_channels_list[3],
                                        stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels_list[3] * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.max_pool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def create_layer(self, ResBlock, blocks, in_channels, out_channels, stride=1):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(ResBlock(in_channels, out_channels, stride=stride, first_block=True))
            else:
                layers.append(ResBlock(out_channels * ResBlock.expansion, out_channels))

        return nn.Sequential(*layers)


def ResNet18(cfg):
    return ResNet(BasicBlock, [2, 2, 2, 2], **cfg)


def ResNet34(cfg):
    return ResNet(BasicBlock, [3, 4, 6, 3], **cfg)


def ResNet50(cfg):
    return ResNet(Bottleneck, [3, 4, 6, 3], **cfg)


def ResNet101(cfg):
    return ResNet(Bottleneck, [3, 4, 23, 3], **cfg)


def ResNet152(cfg):
    return ResNet(Bottleneck, [3, 8, 36, 3], **cfg)


MODELS = {
    "SimpleModel": SimpleModel,
    "ClassifierModel": ClassifierModel,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152
}


def get_model(cfg):
    return MODELS[cfg["network"]](remove_key(cfg, ["network"]))

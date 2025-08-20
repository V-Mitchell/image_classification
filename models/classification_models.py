import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dict import remove_key


class LeNet5(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(LeNet5, self).__init__()
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


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate=12, compression=0.5):
        super(DenseBlock, self).__init__()
        channels = []
        channels.append(in_channels)
        self.relu = nn.ReLU()
        self.layers = []
        for i in range(num_layers):
            # print(i, sum(channels), "->", int(sum(channels) * compression) + growth_rate)
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(sum(channels)), nn.ReLU(),
                    nn.Conv2d(sum(channels), int(sum(channels) * compression), kernel_size=1),
                    nn.BatchNorm2d(int(sum(channels) * compression)), nn.ReLU(),
                    nn.Conv2d(int(sum(channels) * compression),
                              int(sum(channels) * compression) + growth_rate,
                              kernel_size=3,
                              stride=1,
                              padding=1)))
            self.out_channels = int(sum(channels) * compression) + growth_rate
            channels.append(int(sum(channels) * compression) + growth_rate)
            print(i, channels)

        self.num_layers = num_layers
        self.layers = nn.ModuleList(self.layers)

        # self.comp_bn0 = nn.BatchNorm2d(sum(channels))
        # self.comp_conv0 = nn.Conv2d(sum(channels), int(sum(channels) * compression), kernel_size=1)
        # self.bn0 = nn.BatchNorm2d(int(sum(channels) * compression))
        # self.conv0 = nn.Conv2d(int(sum(channels) * compression),
        #                        int(sum(channels) * compression) + growth_rate,
        #                        kernel_size=3,
        #                        stride=1,
        #                        padding=1)
        # channels.append(int(sum(channels) * compression) + growth_rate)

        # print(sum(channels), "->", int(sum(channels) * compression) + growth_rate)
        # self.comp_bn1 = nn.BatchNorm2d(sum(channels))
        # self.comp_conv1 = nn.Conv2d(sum(channels), int(sum(channels) * compression), kernel_size=1)
        # self.bn1 = nn.BatchNorm2d(int(sum(channels) * compression))
        # self.conv1 = nn.Conv2d(int(sum(channels) * compression),
        #                        int(sum(channels) * compression) + growth_rate,
        #                        kernel_size=3,
        #                        stride=1,
        #                        padding=1)
        # channels.append(int(sum(channels) * compression) + growth_rate)

        # print(sum(channels), "->", int(sum(channels) * compression) + growth_rate)
        # self.comp_bn2 = nn.BatchNorm2d(sum(channels))
        # self.comp_conv2 = nn.Conv2d(sum(channels), int(sum(channels) * compression), kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(int(sum(channels) * compression))
        # self.conv2 = nn.Conv2d(int(sum(channels) * compression),
        #                        int(sum(channels) * compression) + growth_rate,
        #                        kernel_size=3,
        #                        stride=1,
        #                        padding=1)
        # channels.append(int(sum(channels) * compression) + growth_rate)

        # print(sum(channels), "->", int(sum(channels) * compression) + growth_rate)
        # self.comp_bn3 = nn.BatchNorm2d(sum(channels))
        # self.comp_conv3 = nn.Conv2d(sum(channels), int(sum(channels) * compression), kernel_size=1)
        # self.bn3 = nn.BatchNorm2d(int(sum(channels) * compression))
        # self.conv3 = nn.Conv2d(int(sum(channels) * compression),
        #                        int(sum(channels) * compression) + growth_rate,
        #                        kernel_size=3,
        #                        stride=1,
        #                        padding=1)
        # channels.append(int(sum(channels) * compression) + growth_rate)

        # print(sum(channels), "->", int(sum(channels) * compression) + growth_rate)
        # self.comp_bn4 = nn.BatchNorm2d(sum(channels))
        # self.comp_conv4 = nn.Conv2d(sum(channels), int(sum(channels) * compression), kernel_size=1)
        # self.bn4 = nn.BatchNorm2d(int(sum(channels) * compression))
        # self.conv4 = nn.Conv2d(int(sum(channels) * compression),
        #                        int(sum(channels) * compression) + growth_rate,
        #                        kernel_size=3,
        #                        stride=1,
        #                        padding=1)

    def forward(self, x):
        outputs = [x]
        for i in range(self.num_layers):
            outputs.append(self.layers[i](torch.cat(outputs, dim=1)))

        # x0 = self.comp_conv0(self.relu(self.comp_bn0(x)))
        # x0 = self.conv0(self.relu(self.bn0(x0)))
        # x1 = self.comp_conv1(self.relu(self.comp_bn1(torch.cat([x, x0], dim=1))))
        # x1 = self.conv1(self.relu(self.bn1(x1)))
        # x2 = self.comp_conv2(self.relu(self.comp_bn2(torch.cat([x, x0, x1], dim=1))))
        # x2 = self.conv2(self.relu(self.bn2(x2)))
        # x3 = self.comp_conv3(self.relu(self.comp_bn3(torch.cat([x, x0, x1, x2], dim=1))))
        # x3 = self.conv3(self.relu(self.bn3(x3)))
        # x4 = self.comp_conv4(self.relu(self.comp_bn4(torch.cat([x, x0, x1, x2, x3], dim=1))))
        # x4 = self.conv4(self.relu(self.bn4(x4)))
        return outputs[-1]


class DenseNet(nn.Module):
    in_channels = 12

    def __init__(self,
                 in_channels=3,
                 num_classes=10,
                 layers_list=[6, 8, 6],
                 growth_rate=12,
                 compression=0.5):
        super(DenseNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels,
                               self.in_channels,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               bias=False)
        self.relu = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(self.in_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = DenseBlock(self.in_channels, layers_list[0], growth_rate, compression)
        self.tran_bn0 = nn.BatchNorm2d(self.layer0.out_channels)
        self.tran_conv0 = nn.Conv2d(self.layer0.out_channels,
                                    int(self.layer0.out_channels * compression),
                                    kernel_size=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.layer1 = DenseBlock(int(self.layer0.out_channels * compression), layers_list[1],
                                 growth_rate, compression)
        self.tran_bn1 = nn.BatchNorm2d(self.layer1.out_channels)
        self.tran_conv1 = nn.Conv2d(self.layer1.out_channels,
                                    int(self.layer1.out_channels * compression),
                                    kernel_size=1)

        self.layer2 = DenseBlock(int(self.layer1.out_channels * compression), layers_list[2],
                                 growth_rate, compression)
        self.tran_bn2 = nn.BatchNorm2d(self.layer2.out_channels)
        self.tran_conv2 = nn.Conv2d(self.layer2.out_channels,
                                    int(self.layer2.out_channels * compression),
                                    kernel_size=1)
        self.tran_bn3 = nn.BatchNorm2d(int(self.layer2.out_channels * compression))
        self.fc = nn.Linear(int(self.layer2.out_channels * compression), num_classes)

    def forward(self, x):
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.max_pool(x)

        x = self.layer0(x)
        x = self.tran_conv0(self.relu(self.tran_bn0(x)))
        x = self.avg_pool(x)

        x = self.layer1(x)
        x = self.tran_conv1(self.relu(self.tran_bn1(x)))
        x = self.avg_pool(x)

        x = self.layer2(x)
        x = self.tran_conv2(self.relu(self.tran_bn2(x)))
        x = self.avg_pool(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def DenseNetX(cfg):
    return DenseNet(**cfg)


class PyramidResNet(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


MODELS = {
    "LeNet5": LeNet5,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "DenseNetX": DenseNetX
}


def get_model(cfg):
    return MODELS[cfg["network"]](remove_key(cfg, ["network"]))


if __name__ == "__main__":
    x = torch.zeros((1, 3, 32, 32), dtype=torch.float)
    print("x", x.shape)
    model = DenseNet()
    y = model(x)
    print("y", y.shape)

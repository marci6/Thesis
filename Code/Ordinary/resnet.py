import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, BatchNorm2d



def conv3x3(in_planes, out_planes, stride=1):
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, args, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1, bias=True)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, args):
        self.inplanes = 32
        super(ResNet, self).__init__()

        self.args = args

        ncha, size, _ = args.inputsize
        self.taskcla = args.taskcla

        self.num_ftrs = 256 * block.expansion

        self.conv1 = Conv2d(ncha, 32, 7, stride=2, padding=3, bias=True)
        
        self.bn1 = BatchNorm2d(32)
        
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

        # self.fc = None
        self.classifier = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.classifier.append(Linear(self.num_ftrs, n))


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=True),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride))

        return nn.Sequential(*layers)

    def add_task(self, num_classes):
        self.classifier = Linear(self.num_ftrs, num_classes)

    def prune(self, mask_modules):
        for module, mask in mask_modules.items():
            module.prune_module(mask)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 3, 2, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)

        y = []
        for t, i in self.taskcla:
            y.append(self.classifier[t](x))
        return [F.log_softmax(yy, dim=1) for yy in y]


def Net(args):
    return ResNet(BasicBlock, [3, 4, 6, 3], args)
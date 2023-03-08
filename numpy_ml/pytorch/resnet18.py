import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, kernal_size=3):
        super(BasicBlock, self).__init__()
        self.kernal_size = kernal_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, self.kernal_size, self.stride, padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, self.kernal_size, self.stride, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        out = self.bn2(x1)
        return F.relu(x+out)


class BasicBlock2(nn.Module):
    def __init__(self, in_channel, out_channel, stride:list, kernal_size=3):
        super(BasicBlock2, self).__init__()
        self.kernal_size = kernal_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, self.kernal_size, self.stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, self.kernal_size, self.stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_channel)
        self.conv3 = nn.Conv2d(self.in_channel, self.out_channel, stride = self.stride[0], kernal_size=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.out_channel)



    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        out = self.bn2(x1)
        x2 = self.conv3(x)
        x2 = self.bn2(x2)
        return F.relu(x2+out)




class Resnet18(nn.Module):
    def __init__(self, class_num):
        super(Resnet18).__init__()
        self.conv1 = nn.Conv2d(3, 64, stride=2, padding=3, kernel_size=7)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, padding=3, stride=2)
        self.layer1 = nn.Sequential(BasicBlock(64,64,1), BasicBlock(64,64,1))
        self.layer2 = nn.Sequential(BasicBlock2(64,128,[2,1]), BasicBlock(128,128,1))
        self.layer3 = nn.Sequential(BasicBlock2(128,256,[2,1]), BasicBlock(256,256,1))
        self.layer4 = nn.Sequential(BasicBlock2(256,512,[2,1]), BasicBlock(512,512,1))
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512,class_num)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool2(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out






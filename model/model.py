import torch
from torch import nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        padding: int, 
        bnorm: bool = False
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if bnorm else None 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        return F.relu(x)

class InceptionModule(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        ch1x1: int, 
        ch3x3red: int, 
        ch3x3: int, 
        ch5x5red: int, 
        ch5x5: int, 
        pool_proj: int,
        bnorm: bool = False
    ) -> None:
        super().__init__()
        self.branch1 = Conv(in_channels, ch1x1, kernel_size=1, stride=1, padding=0, bnorm=bnorm)
        self.branch2 = nn.Sequential(
            Conv(in_channels, ch3x3red, kernel_size=1, stride=1, padding=0, bnorm=bnorm),
            Conv(ch3x3red, ch3x3, kernel_size=3, stride=1, padding=1, bnorm=bnorm)
        )
        self.branch3 = nn.Sequential(
            Conv(in_channels, ch5x5red, kernel_size=1, stride=1, padding=0, bnorm=bnorm),
            Conv(ch5x5red, ch5x5, kernel_size=3, stride=1, padding=1, bnorm=bnorm)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
            Conv(in_channels, pool_proj, kernel_size=1, stride=1, padding=0, bnorm=bnorm)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch1(x),
            self.branch2(x), 
            self.branch3(x), 
            self.branch4(x)
        ], dim=1)

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, bnorm: bool = True) -> None:
        super().__init__()
        self.bnorm = bnorm
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = Conv(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(128*4*4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if not self.bnorm: x = self.dropout(x)
        x = self.fc2(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, bnorm: bool = False, aux_flag: bool = True):
        super().__init__()
        self.bnorm = bnorm

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32, self.bnorm)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64, self.bnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64, self.bnorm)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64, self.bnorm)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64, self.bnorm)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64, self.bnorm)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128, self.bnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        if aux_flag: 
            self.aux4a = AuxiliaryClassifier(512, num_classes, bnorm=self.bnorm)
            self.aux4d = AuxiliaryClassifier(528, num_classes, bnorm=self.bnorm)
        else: 
            self.aux4a, self.aux4d = None, None

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128, self.bnorm)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128, self.bnorm)
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)
        if not self.bnorm: x = F.local_response_norm(x, 5)
        
        x = self.conv2(x)
        x = self.conv3(x)
        if not self.bnorm: x = F.local_response_norm(x, 5)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        aux4a = self.aux4a(x) if self.aux4a is not None and self.training else None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux4d = self.aux4d(x) if self.aux4d is not None and self.training else None
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        if not self.bnorm: x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        
        return x, aux4d, aux4a
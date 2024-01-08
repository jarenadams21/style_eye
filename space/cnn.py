import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

# ----------------------------
# Image Classification Model
# ----------------------------
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution Block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        # Convolution Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)

        # Convolution Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.4)

        # Adaptive pooling and Linear Layer
        self.ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.lin = nn.Linear(64, 27)

        # Initialize weights using Kaiming Normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.drop3(self.relu3(self.bn3(self.conv3(x))))
        x = self.ap(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return x
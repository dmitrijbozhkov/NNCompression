import torch.nn as nn
import torch.nn.functional as F
import torch


# 496 weights
class LeNet3(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5, padding=0, stride=2)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=0)
        self.fc1 = nn.Linear(6 * 2 * 2, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


# 1306 weights
class LeNet3_2(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet3_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0, stride=2)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=0)
        self.fc1 = nn.Linear(12 * 2 * 2, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


# 2026 weights
class LeNet3_3(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet3_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=0, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=0)
        self.fc1 = nn.Linear(16 * 2 * 2, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


class LeNet3_4(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet3_4, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=0, stride=2)
        self.conv2 = nn.Conv2d(16, 48, 3, padding=0)
        self.fc1 = nn.Linear(48 * 2 * 2, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


class LeNet4(nn.Module):
    # 4834 weights
    def __init__(self, num_classes=10):
        super(LeNet4, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=0)  # 24 24 -->12 12
        self.conv2 = nn.Conv2d(6, 12, 5, padding=0)  # 8 8 -->4 4
        self.conv3 = nn.Conv2d(12, 24, 3, padding=0)
        self.fc1 = nn.Linear(24, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 48, 3, padding=1)
        self.conv3 = nn.Conv2d(48, 96, 3, padding=1)
        self.fc1 = nn.Linear(96 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

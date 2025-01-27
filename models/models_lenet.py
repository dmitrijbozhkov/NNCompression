import torch.nn as nn
import torch.nn.functional as F
import torch


class QuantModel(nn.Module):
    """Common methods for quntization model"""

    def __init__(self):
        super(QuantModel, self).__init__()
        self.original_weights = {}


    def permute_weights(self, perturbations):
        for name, param in self.named_parameters():
            if name in perturbations:
                self.original_weights[name] = param.data.clone()  # Store  original weights
                param.data.add_(perturbations[name])  # Add perturbation

    def change_back_weights(self):
        for name, param in self.named_parameters():
            if name in self.original_weights:
                param.data.copy_(self.original_weights[name])


# 496 weights
class LeNet3(QuantModel):

    def __init__(self):
        super(LeNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5, padding=0, stride=2)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=0)
        self.fc1 = nn.Linear(6 * 2 * 2, 10)

    def forward(self, x, permutations=None):
        if permutations:
            self.permute_weights(permutations)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        if permutations:
            self.change_back_weights()
        return x


# 1306 weights
class LeNet3_2(QuantModel):
    def __init__(self):
        super(LeNet3_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0, stride=2)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=0)
        self.fc1 = nn.Linear(12 * 2 * 2, 10)

    def forward(self, x, permutations=None):
        if permutations:
            self.permute_weights(permutations)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        if permutations:
            self.change_back_weights()
        return x


# 2026 weights
class LeNet3_3(QuantModel):
    def __init__(self):
        super(LeNet3_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=0, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=0)
        self.fc1 = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x, permutations=None):
        if permutations:
            self.permute_weights(permutations)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        if permutations:
            self.change_back_weights()
        return x


class LeNet3_4(QuantModel):
    def __init__(self):
        super(LeNet3_4, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=0, stride=2)
        self.conv2 = nn.Conv2d(16, 48, 3, padding=0)
        self.fc1 = nn.Linear(48 * 2 * 2, 10)

    def forward(self, x, permutations=None):
        if permutations:
            self.permute_weights(permutations)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        if permutations:
            self.change_back_weights()
        return x


class LeNet_4(QuantModel):
    # 4834 weights
    def __init__(self):
        super(LeNet_4, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=0)  # 24 24 -->12 12
        self.conv2 = nn.Conv2d(6, 12, 5, padding=0)  # 8 8 -->4 4
        self.conv3 = nn.Conv2d(12, 24, 3, padding=0)
        self.fc1 = nn.Linear(24, 10)

    def forward(self, x, permutations=None):
        if permutations:
            self.permute_weights(permutations)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        if permutations:
            self.change_back_weights()
        return x


class LeNet_5(QuantModel):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 48, 3, padding=1)
        self.conv3 = nn.Conv2d(48, 96, 3, padding=1)
        self.fc1 = nn.Linear(96 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, permutations=None):
        if permutations:
            self.permute_weights(permutations)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if permutations:
            self.change_back_weights()
        return x


# 80000 weights
class Model_on_cifar10_2(QuantModel):
    def __init__(self):
        super(Model_on_cifar10_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 3, padding=1)
        self.fc1 = nn.Linear(48 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x, permutations=None):
        if permutations:
            self.permute_weights(permutations)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if permutations:
            self.change_back_weights()
        return x


class Model_on_cifar10(QuantModel):
    def __init__(self):
        super(Model_on_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 48, 3, padding=1)
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x, permutations=None):
        if permutations:
            self.permute_weights(permutations)
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=(2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if permutations:
            self.change_back_weights()
        return x


class AlexNet(QuantModel):

    def __init__(self, classes=100):
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
            nn.Linear(4096, classes),
        )

    def forward(self, x, permutations=None):
        if permutations:
            self.permute_weights(permutations)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if permutations:
            self.change_back_weights()
        return x

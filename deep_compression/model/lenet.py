from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def total_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())


class LeNet5(LeNet):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x shape: [batch_size, 1, 32, 32]
        x = self.relu(self.conv1(x))  # [batch_size, 6, 28, 28]
        x = self.pool1(x)  # [batch_size, 6, 14, 14]
        x = self.relu(self.conv2(x))  # [batch_size, 16, 10, 10]
        x = self.pool2(x)  # [batch_size, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5)  # Flatten [batch_size, 400]
        x = self.relu(self.fc1(x))  # [batch_size, 120]
        x = self.relu(self.fc2(x))  # [batch_size, 84]
        x = self.fc3(x)  # [batch_size, num_classes]
        return x


class LeNet300100(LeNet):
    def __init__(self, num_classes=10):
        super(LeNet300100, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x shape: [batch_size, 1, 28, 28]
        x = x.view(-1, 28 * 28)  # [batch_size, 784]
        x = self.relu(self.fc1(x))  # [batch_size, 300]
        x = self.relu(self.fc2(x))  # [batch_size, 100]
        x = self.fc3(x)  # [batch_size, num_classes]
        return x

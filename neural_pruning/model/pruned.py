from typing import Any, override

import torch
from torch import nn

from .prunable import PrunableModel


class PrunedModel(PrunableModel):
    def __init__(self):
        super().__init__()

    @property
    def prune_thresholds(self) -> dict[str, float]:
        """Heuristic to determine layer-wise pruning thresholds"""
        thresholds = {}

        for name, param in self.named_parameters():
            non_zero_part = param.data[param.data != 0].abs().view(-1)
            quantile_dict = {
                "conv1.weight": 0.2,
                "conv2.weight": 0.35,
                "fc1.weight": 0.5,
                "fc2.weight": 0.6,
                "fc3.weight": 0.1,
            }
            quantile = quantile_dict.get(name, 0.0)

            if non_zero_part.numel() == 0:
                threshold = torch.Tensor([0.0])
            else:
                threshold = torch.quantile(non_zero_part, quantile)

            thresholds[name] = threshold.item()

        return thresholds

    @override
    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        for name in self.masks:
            state_dict[f"{name}.mask"] = self.masks[name]
        return state_dict

    @override
    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key.endswith(".mask"):
                param_name = key[:-5]
                self.masks[param_name] = value

        return super().load_state_dict(state_dict, strict=False)

    def init_masks(self):
        for name, param in self.named_parameters():
            self.masks[name] = torch.ones_like(param, dtype=torch.bool)


# Redundant pruned model definitions here, but I currently think this is the best way to keep the code organized.
# Since if I use inheritance, diamond inheritance problem will occur.
# and if I use composition, the model definition will be clumsy.
class PrunedLeNet5(PrunedModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.init_masks()

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

    @property
    def name(self) -> str:
        return "PrunedLeNet5"


class PrunedLeNet300100(PrunedModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()
        self.init_masks()

    def forward(self, x):
        # Input x shape: [batch_size, 1, 28, 28]
        x = x.view(-1, 28 * 28)  # [batch_size, 784]
        x = self.relu(self.fc1(x))  # [batch_size, 300]
        x = self.relu(self.fc2(x))  # [batch_size, 100]
        x = self.fc3(x)  # [batch_size, num_classes]
        return x

    @property
    def name(self) -> str:
        return "PrunedLeNet300100"

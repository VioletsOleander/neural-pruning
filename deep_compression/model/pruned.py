from collections.abc import Iterator

import torch
from torch import nn


class PrunedModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.masks: dict[str, torch.Tensor] = {}

        # Initialize masks for all prunable parameters
        for name, param in self.base_model.named_parameters():
            if "bias" not in name:
                self.masks[name] = torch.ones_like(param, dtype=torch.bool)

    def forward(self, x):
        return self.base_model(x)

    def named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        return self.base_model.named_parameters()

    def parameters(self) -> Iterator[torch.Tensor]:
        return self.base_model.parameters()

    def apply_pruning(self):
        """Apply stored masks to parameters"""
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in self.masks:
                    param.mul_(self.masks[name].float())

    def apply_gradient_masking(self):
        """Apply masks to gradients"""
        for name, param in self.base_model.named_parameters():
            if name in self.masks and param.grad is not None:
                param.grad.mul_(self.masks[name].float())

    def state_dict(self):
        state_dict = self.base_model.state_dict()
        for name in self.masks:
            state_dict[f"{name}.mask"] = self.masks[name]
        return state_dict

    def load_state_dict(self, state_dict):
        base_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".mask"):
                param_name = key[:-5]
                self.masks[param_name] = value
            else:
                base_state_dict[key] = value
        return self.base_model.load_state_dict(base_state_dict)

    def get_prune_thresholds(self) -> dict[str, float]:
        """Heuristic to determine layer-wise pruning thresholds"""
        thresholds = {}

        for name, param in self.base_model.named_parameters():
            if "bias" in name:
                thresholds[name] = 0.0
                continue

            non_zero_weights = param.data[param.data != 0].abs().view(-1)

            if "conv1" in name:
                quantile = 0.2
            elif "conv2" in name:
                quantile = 0.35
            elif "fc1" in name:
                quantile = 0.5
            elif "fc2" in name:
                quantile = 0.6
            elif "fc3" in name:
                quantile = 0.1
            else:
                quantile = 0.4

            threshold = (
                torch.quantile(non_zero_weights, quantile)
                if non_zero_weights.numel() > 0
                else torch.tensor(0.0)
            )
            thresholds[name] = threshold.item()

        return thresholds

    def parameter_count(self) -> int:
        """Count effective (non-pruned) parameters"""
        total = 0
        for name, param in self.base_model.named_parameters():
            if name in self.masks:
                total += self.masks[name].sum().item()
            else:
                total += param.numel()
        return int(total)

    def total_bytes(self) -> int:
        """Calculate effective (non-pruned) model size in bytes"""
        total = 0
        for name, param in self.base_model.named_parameters():
            if name in self.masks:
                total += self.masks[name].sum().item() * param.element_size()
            else:
                total += param.numel() * param.element_size()
        return int(total)

    def __repr__(self) -> str:
        return f"PrunedModel(base_model={self.base_model})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def name(self) -> str:
        return f"Pruned{self.base_model.__class__.__name__}"

    @property
    def sparsity(self) -> float:
        """Calculate overall sparsity of the model"""
        total_params = sum(p.numel() for p in self.base_model.parameters())
        if total_params == 0:
            return 0.0

        effective_params = self.parameter_count()
        return (total_params - effective_params) / total_params


if __name__ == "__main__":
    from deep_compression.model.lenet import LeNet5

    base_model = LeNet5()
    pruned_model = PrunedModel(base_model)

    print(pruned_model)

    thresholds = pruned_model.get_prune_thresholds()
    print("Prune Thresholds:", thresholds)

    print(
        pruned_model.name,
        "has",
        pruned_model.parameter_count(),
        "parameters.",
    )

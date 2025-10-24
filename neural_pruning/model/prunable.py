from typing import Optional, override

import torch
from torch import nn

from ._counter import Counter


def _compute_flops(
    module: nn.Module, output: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> int:
    """Compute effective FLOPs for a module, accounting for pruning masks and batch size"""
    batch_size = output.shape[0]

    # Handle convolutional layers
    if isinstance(module, nn.Conv2d):
        out_spatial_size = output.shape[2] * output.shape[3]
        out_channels = output.shape[1]

        # Count number of non-pruned weights
        if mask is not None:
            total_non_pruned = torch.sum(mask).item()
        else:
            total_non_pruned = module.weight.numel()

        # Conv FLOPs = 2 ops per weight * spatial positions * batch size
        flops = 2 * total_non_pruned * out_spatial_size * batch_size

        # Bias FLOPs = 1 ops per weight * out_channels * spatial positions * batch sizes
        if module.bias is not None:
            flops += out_channels * out_spatial_size * batch_size

    # Handle linear layers
    elif isinstance(module, nn.Linear):
        if mask is not None:
            total_non_pruned = torch.sum(mask).item()
        else:
            total_non_pruned = module.weight.numel()

        # Linear FLOPs = 2 ops per weight * batch size
        flops = 2 * total_non_pruned * batch_size

        # Bias FLOPs = 1 ops per weight * out_features * batch size
        if module.bias is not None:
            flops += module.out_features * batch_size

    # Other layers are ignored for FLOPs calculation
    else:
        flops = 0

    return int(flops)


class PrunableModel(nn.Module, Counter):
    def __init__(self):
        super().__init__()
        self.masks: dict[str, torch.Tensor] = {}

    @property
    def sparsity(self) -> float:
        """Calculate overall sparsity of the model"""
        if not self.masks:
            return 0.0

        total_params = sum(p.numel() for p in self.parameters())
        if total_params == 0:
            return 0.0
        effective_params = self.total_parameters()
        return (total_params - effective_params) / total_params

    @override
    def per_layer_parameters(self) -> dict:
        """Calculate effective parameter count per layer"""
        param_count_dict = {}
        for name, param in self.named_parameters():
            if name in self.masks:
                param_count_dict[name] = torch.sum(self.masks[name]).item()
            else:
                param_count_dict[name] = param.numel()
        return param_count_dict

    @override
    def per_layer_bytes(self) -> dict:
        """Calculate effective model size (bytes) per layer"""
        bytes_count_dict = {}
        for name, param in self.named_parameters():
            if name in self.masks:
                bytes_count_dict[name] = (
                    torch.sum(self.masks[name]).item() * param.element_size()
                )
            else:
                bytes_count_dict[name] = param.numel() * param.element_size()
        return bytes_count_dict

    @override
    def per_layer_flops(self, input_size) -> dict[str, int]:
        """Calculate effective FLOPs per layer"""
        flops_count_dict = {}
        hooks = []

        # Closure to capture mask and module name
        def _make_hook(module_name: str):
            weight_name = f"{module_name}.weight" if module_name else "weight"
            mask = self.masks.get(weight_name, None)

            def _hook_fn(module, input, output):
                flops = _compute_flops(module, output, mask)
                flops_count_dict[module_name] = flops

            return _hook_fn

        # Register hooks for all leaf modules
        for name, module in self.named_modules():
            if not list(module.children()):
                hook = _make_hook(name)
                hooks.append(module.register_forward_hook(hook))

        # Forward
        x = torch.zeros(input_size)
        self.eval()
        with torch.no_grad():
            self(x)

        # Remove all hooks
        for hook in hooks:
            hook.remove()

        return flops_count_dict

    def prune_parameters(self):
        """Prune parameters by applying masks in-place. For unpruned models, this is a no-op."""
        if not self.masks:
            return

        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.masks:
                    param.mul_(self.masks[name].float())

    def mask_gradients(self):
        """Mask gradients during backpropagation to prevent updates to pruned parameters. For unpruned models, this is a no-op."""
        if not self.masks:
            return

        for name, param in self.named_parameters():
            if name in self.masks and param.grad is not None:
                param.grad.mul_(self.masks[name].float())

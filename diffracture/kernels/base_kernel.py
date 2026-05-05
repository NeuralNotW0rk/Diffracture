from torch import Tensor
import torch.nn as nn

from ..topology.base_prism import Prism


class Kernel:
    # Descriptive mathematical properties of the kernel.
    # Encapsulated in a dictionary so systems can query them dynamically.
    traits = {
        "is_additive": False,           # O(1) activation math, ideal for hooks
        "requires_weight_fusion": False # Modifies base weights, requires parametrization/grafting
    }

    def __call__(self, x: Tensor, prism: Prism, original_module: nn.Module):
        """Standard execution during the forward pass."""
        raise NotImplementedError

    def forward_delta(self, x: Tensor, base_output: Tensor, prism: Prism, original_module: nn.Module) -> Tensor:
        """Computes only the modification to the activations (used by HookInjector)."""
        raise NotImplementedError

    def is_supported(self, original_module: nn.Module) -> bool:
        """Returns True if the kernel can operate on the given module."""
        raise NotImplementedError

    def compute_delta(self, prism: Prism) -> Tensor:
        """Required for the DirectInjector / merging logic."""
        raise NotImplementedError

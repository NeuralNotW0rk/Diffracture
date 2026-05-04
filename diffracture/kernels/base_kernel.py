from torch import Tensor
import torch.nn as nn

from ..topology.base_prism import Prism


class Kernel:
    def __call__(self, x: Tensor, prism: Prism, original_module: nn.Module):
        """Standard execution during the forward pass."""
        raise NotImplementedError

    def is_supported(self, original_module: nn.Module) -> bool:
        """Returns True if the kernel can operate on the given module."""
        raise NotImplementedError

    def compute_delta(self, prism: Prism) -> Tensor:
        """Required for the DirectInjector / merging logic."""
        raise NotImplementedError

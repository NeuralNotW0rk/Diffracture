from torch import Tensor
import torch.nn as nn

from ..topology.base_element import Element


class Kernel:
    # Descriptive mathematical properties of the kernel.
    # Encapsulated in a dictionary so systems can query them dynamically.
    traits = {
        "is_additive": False,           # O(1) activation math, ideal for hooks
        "requires_weight_fusion": False # Modifies base weights, requires parametrization/grafting
    }

    def __call__(self, x: Tensor, element: Element, original_module: nn.Module):
        """Standard execution during the forward pass."""
        raise NotImplementedError

    def forward_delta(self, x: Tensor, base_output: Tensor, element: Element, original_module: nn.Module) -> Tensor:
        """Computes only the modification to the activations (used by HookInjector)."""
        raise NotImplementedError

    def is_supported(self, original_module: nn.Module) -> bool:
        """Returns True if the kernel can operate on the given module."""
        raise NotImplementedError

    def supports_activation(self, original_module: nn.Module, element: Element) -> bool:
        """
        Returns True if forward_delta / __call__ are valid for this element on this module, i.e. the
        kernel knows the module's forward well enough to work in activation space (hooks, grafts).
        """
        return self.is_supported(original_module)

    def supports_weights(self, original_module: nn.Module, element: Element) -> bool:
        """
        Returns True if compute_delta is valid for this element on this module. Weight-space kernels
        need nothing from the module but the tensor they modify, so they apply to any module whose
        forward reads that tensor (parametrization, direct merges).
        """
        return False

    def compute_delta(self, element: Element) -> Tensor:
        """Required for the DirectInjector / merging logic."""
        raise NotImplementedError

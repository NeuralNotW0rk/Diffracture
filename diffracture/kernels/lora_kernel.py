import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_kernel import Kernel
from ..registry import register_kernel

# The Dispatch Registry
LORA_HANDLERS = {
    nn.Linear: lambda x, w, b, m: F.linear(x, w, b),
    nn.Conv1d: lambda x, w, b, m: F.conv1d(
        x, w, b, 
        stride=m.stride, 
        padding=m.padding, 
        dilation=m.dilation, 
        groups=m.groups
    ),
    nn.Conv2d: lambda x, w, b, m: F.conv2d(
        x, w, b, 
        stride=m.stride, 
        padding=m.padding, 
        dilation=m.dilation, 
        groups=m.groups
    )
}

@register_kernel("lora")
class LoRAKernel(Kernel):
    """
    Executes standard Low-Rank Adaptation (LoRA).
    Because standard LoRA is purely additive, this kernel calculates the delta 
    independently and is highly optimized for use with the HookInjector.
    """
    traits = {
        "is_additive": True,
        "requires_weight_fusion": False
    }

    def is_supported(self, original_module: nn.Module) -> bool:
        return type(original_module) in LORA_HANDLERS

    def forward_delta(self, x: Tensor, base_output: Tensor, element, original_module) -> Tensor:
        """Computes the activation delta without triggering original_module."""
        handler = LORA_HANDLERS.get(type(original_module))
        if not handler:
            raise TypeError(f"Unhandled module type: {type(original_module)}")

        # Parameters from Element
        lora_a = element.params["lora_down"].to(x.device, dtype=x.dtype)
        lora_b = element.params["lora_up"].to(x.device, dtype=x.dtype)
        rank = element.metadata.get("rank", lora_a.size(0))
        alpha = element.metadata.get("alpha", rank)
        strength = getattr(element, "strength", 1.0)
        scale = (alpha / rank) * strength

        # LoRA Path
        # Using the lambda handlers to normalize the different F signatures
        lx = handler(x, lora_a, None, original_module)
        lx = handler(lx, lora_b, None, original_module)

        return lx * scale

    def __call__(self, x: Tensor, element, original_module):
        base_output = original_module(x)
        return base_output + self.forward_delta(x, base_output, element, original_module)
    
    def compute_delta(self, element, original_module) -> dict:
        # Standard LoRA math
        weight_a = element.params["lora_down"].view(element.params["lora_down"].size(0), -1)
        weight_b = element.params["lora_up"].view(element.params["lora_up"].size(0), -1)
        
        rank = element.metadata.get("rank", weight_a.size(0))
        alpha = element.metadata.get("alpha", rank)
        strength = getattr(element, "strength", 1.0)
        scale = (alpha / rank) * strength
        weight_delta = (weight_b @ weight_a) * scale
        weight_delta = weight_delta.view(original_module.weight.shape)
        
        w_orig = original_module.weight
        weight_delta = weight_delta.to(device=w_orig.device, dtype=w_orig.dtype)
        
        # Return a map of updates
        updates = {
            "weight": weight_delta
        }
        
        # Offload the bias check here
        if "bias" in element.params:
            updates["bias"] = element.params["bias"]
            
        return updates
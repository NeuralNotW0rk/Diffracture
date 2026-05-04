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
    Executes Low-Rank Adaptation (LoRA) and its direct variants (e.g., DoRA).
    
    If a technique alters the flavor of LoRA (e.g., changing initialization, scaling, 
    or adding a magnitude vector like DoRA) but retains the fundamental (B @ A) 
    linear algebra structure, it should be handled via metadata flags and executed within this kernel.
    
    If a technique fundamentally changes the topological structure or matrix operations 
    (e.g., Kronecker products in LoKr, or Hadamard products in LoHa), it warrants an entirely new Kernel.
    """
    def is_supported(self, original_module: nn.Module) -> bool:
        return type(original_module) in LORA_HANDLERS

    def __call__(self, x: Tensor, prism, original_module):
        # Base pass
        base_output = original_module(x)

        # Dynamic Lookup
        handler = LORA_HANDLERS.get(type(original_module))
        if not handler:
            raise TypeError(f"Unhandled module type: {type(original_module)}")

        # DoRA Path (Materializes fused weight dynamically)
        if "magnitude" in prism.params:
            deltas = self.compute_delta(prism, original_module)
            fused_weight = original_module.weight + deltas["weight"]
            fused_weight = fused_weight.to(dtype=x.dtype)
            
            # Maintain the original bias unless prism overrides it
            bias = deltas.get("bias", original_module.bias)
            if bias is not None: bias = bias.to(dtype=x.dtype)
            return handler(x, fused_weight, bias, original_module)

        # Parameters from Prism
        lora_a = prism.params["lora_down"].to(x.device, dtype=x.dtype)
        lora_b = prism.params["lora_up"].to(x.device, dtype=x.dtype)
        rank = prism.metadata.get("rank", lora_a.size(0))
        alpha = prism.metadata.get("alpha", rank)
        scale = (alpha / rank) * prism.multiplier

        # LoRA Path
        # Using the lambda handlers to normalize the different F signatures
        lx = handler(x, lora_a, None, original_module)
        lx = handler(lx, lora_b, None, original_module)

        return base_output + (lx * scale)
    
    def compute_delta(self, prism, original_module) -> dict:
        # Standard LoRA math
        weight_a = prism.params["lora_down"].view(prism.params["lora_down"].size(0), -1)
        weight_b = prism.params["lora_up"].view(prism.params["lora_up"].size(0), -1)
        
        rank = prism.metadata.get("rank", weight_a.size(0))
        alpha = prism.metadata.get("alpha", rank)
        scale = (alpha / rank) * prism.multiplier
        weight_delta = (weight_b @ weight_a) * scale
        weight_delta = weight_delta.view(original_module.weight.shape)
        
        w_orig = original_module.weight
        weight_delta = weight_delta.to(device=w_orig.device, dtype=w_orig.dtype)
        
        if "magnitude" in prism.params:
            w_fused = w_orig + weight_delta
            
            mag = prism.params["magnitude"].to(device=w_orig.device, dtype=w_orig.dtype)
            norm_dims = list(range(1, w_fused.dim()))
            norm = w_fused.norm(p=2, dim=norm_dims, keepdim=True)
            
            mag_view_shape = [mag.size(0)] + [1] * (w_fused.dim() - 1)
            mag_view = mag.view(*mag_view_shape)
            
            dora_weight = w_fused * (mag_view / norm)
            weight_delta = dora_weight - w_orig
        
        # Return a map of updates
        updates = {
            "weight": weight_delta
        }
        
        # Offload the bias check here
        if "bias" in prism.params:
            updates["bias"] = prism.params["bias"]
            
        return updates
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from base_kernel import Kernel

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

class LoRAKernel(Kernel):
    def __call__(self, x: Tensor, prism, original_module, registry=LORA_HANDLERS):
        # Base pass
        base_output = original_module(x)

        # Dynamic Lookup
        handler = registry.get(type(original_module))
        if not handler:
            raise TypeError(f"Unhandled module type: {type(original_module)}")

        # Parameters from Prism
        lora_a = prism.params["lora_down"].to(x.device, dtype=x.dtype)
        lora_b = prism.params["lora_up"].to(x.device, dtype=x.dtype)
        rank = prism.metadata.get("rank", lora_a.size(0))
        alpha = prism.metadata.get("alpha", rank)
        scale = alpha / rank

        # LoRA Path
        # Using the lambda handlers to normalize the different F signatures
        lx = handler(x, lora_a, None, original_module)
        lx = handler(lx, lora_b, None, original_module)

        return base_output + (lx * scale)
    
    def compute_delta(self, prism, original_module) -> dict:
        # Standard LoRA math
        weight_a = prism.params["lora_down"].view(prism.params["lora_down"].size(0), -1)
        weight_b = prism.params["lora_up"].view(prism.params["lora_up"].size(0), -1)
        
        scale = prism.metadata["alpha"] / prism.metadata["rank"]
        weight_delta = (weight_b @ weight_a) * scale
        
        # Return a map of updates
        updates = {
            "weight": weight_delta.view(original_module.weight.shape)
        }
        
        # Offload the bias check here
        if "bias" in prism.params:
            updates["bias"] = prism.params["bias"]
            
        return updates
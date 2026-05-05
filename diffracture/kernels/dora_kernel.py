import torch.nn as nn
from torch import Tensor

from .base_kernel import Kernel
from ..registry import register_kernel
from .lora_kernel import LORA_HANDLERS

@register_kernel("dora")
class DoRAKernel(Kernel):
    """
    Executes Weight-Decomposed Low-Rank Adaptation (DoRA).
    Because DoRA calculates a dynamically fused and normalized weight matrix 
    rather than producing an additive delta, it is highly recommended to be 
    injected via the GraftInjector to avoid re-evaluating the original module.
    """
    traits = {
        "is_additive": False,
        "requires_weight_fusion": True
    }

    def is_supported(self, original_module: nn.Module) -> bool:
        return type(original_module) in LORA_HANDLERS

    def forward_delta(self, x: Tensor, base_output: Tensor, prism, original_module) -> Tensor:
        """
        Technically computes the activation delta. Note: For DoRA, this is highly 
        inefficient because it requires computing the fused pass entirely and subtracting 
        the base_output. Use GraftInjector to bypass this double evaluation instead.
        """
        fused_output = self(x, prism, original_module)
        return fused_output - base_output

    def __call__(self, x: Tensor, prism, original_module):
        handler = LORA_HANDLERS.get(type(original_module))
        if not handler:
            raise TypeError(f"Unhandled module type: {type(original_module)}")

        deltas = self.compute_delta(prism, original_module)
        fused_weight = original_module.weight + deltas["weight"]
        fused_weight = fused_weight.to(dtype=x.dtype)
        
        bias = deltas.get("bias", original_module.bias)
        if bias is not None: bias = bias.to(dtype=x.dtype)
        
        return handler(x, fused_weight, bias, original_module)
    
    def compute_delta(self, prism, original_module) -> dict:
        weight_a = prism.params["lora_down"].view(prism.params["lora_down"].size(0), -1)
        weight_b = prism.params["lora_up"].view(prism.params["lora_up"].size(0), -1)
        
        rank = prism.metadata.get("rank", weight_a.size(0))
        alpha = prism.metadata.get("alpha", rank)
        scale = (alpha / rank) * prism.multiplier
        weight_delta = (weight_b @ weight_a) * scale
        weight_delta = weight_delta.view(original_module.weight.shape)
        
        w_orig = original_module.weight
        weight_delta = weight_delta.to(device=w_orig.device, dtype=w_orig.dtype)
        
        w_fused = w_orig + weight_delta
        
        mag = prism.params["magnitude"].to(device=w_orig.device, dtype=w_orig.dtype)
        norm_dims = list(range(1, w_fused.dim()))
        norm = w_fused.norm(p=2, dim=norm_dims, keepdim=True)
        
        mag_view_shape = [mag.size(0)] + [1] * (w_fused.dim() - 1)
        mag_view = mag.view(*mag_view_shape)
        
        dora_weight = w_fused * (mag_view / norm)
        weight_delta = dora_weight - w_orig
        
        updates = {"weight": weight_delta}
        
        if "bias" in prism.params:
            updates["bias"] = prism.params["bias"]
            
        return updates
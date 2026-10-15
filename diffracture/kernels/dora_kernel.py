import torch.nn as nn
from torch import Tensor

from .base_kernel import Kernel
from ..registry import register_kernel
from .lora_kernel import (
    LORA_HANDLERS,
    target_param,
    lora_delta_matrix,
    has_lora_activation_path,
    has_lora_weight_target,
)
from ..topology.layout import WeightLayout

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

    def supports_activation(self, original_module: nn.Module, element) -> bool:
        return has_lora_activation_path(original_module, element)

    def supports_weights(self, original_module: nn.Module, element) -> bool:
        return has_lora_weight_target(original_module, element)

    def forward_delta(self, x: Tensor, base_output: Tensor, element, original_module) -> Tensor:
        """
        Technically computes the activation delta. Note: For DoRA, this is highly 
        inefficient because it requires computing the fused pass entirely and subtracting 
        the base_output. Use GraftInjector to bypass this double evaluation instead.
        """
        fused_output = self(x, element, original_module)
        return fused_output - base_output

    def __call__(self, x: Tensor, element, original_module):
        handler = LORA_HANDLERS.get(type(original_module))
        if not handler:
            raise TypeError(f"Unhandled module type: {type(original_module)}")

        deltas = self.compute_delta(element, original_module)
        fused_weight = original_module.weight + deltas["weight"]
        fused_weight = fused_weight.to(dtype=x.dtype)
        
        bias = deltas.get("bias", original_module.bias)
        if bias is not None: bias = bias.to(dtype=x.dtype)
        
        return handler(x, fused_weight, bias, original_module)
    
    def compute_delta(self, element, original_module) -> dict:
        param = target_param(element)
        w_orig = getattr(original_module, param)
        layout = WeightLayout.from_metadata(element.metadata.get("layout"))

        weight_delta = lora_delta_matrix(element).to(device=w_orig.device, dtype=w_orig.dtype)
        w_fused = layout.to_matrix(w_orig) + weight_delta

        # Each output's weights (a row of the [out, in] matrix) are renormalized to the learned magnitude
        mag = element.params["magnitude"].to(device=w_orig.device, dtype=w_orig.dtype)
        norm = w_fused.norm(p=2, dim=1)

        dora_weight = w_fused * (mag / norm).unsqueeze(1)
        weight_delta = layout.from_matrix(dora_weight, w_orig.shape) - w_orig

        updates = {param: weight_delta}
        
        if "bias" in element.params:
            updates["bias"] = element.params["bias"]
            
        return updates

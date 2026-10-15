import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_kernel import Kernel
from ..registry import register_kernel
from ..topology.layout import WeightLayout

# The Dispatch Registry
# Activation-space fast paths for modules whose forward is known exactly. Matched by exact type:
# a subclass may override forward (or a parametrized module may wrap it), so it takes the weight path.
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


def target_param(element) -> str:
    """Name of the module parameter the element adapts."""
    return element.metadata.get("param") or "weight"


def lora_scale(element) -> float:
    rank = element.metadata.get("rank", element.params["lora_down"].size(0))
    alpha = element.metadata.get("alpha", rank)
    strength = getattr(element, "strength", 1.0)
    reference_scale = element.metadata.get("reference_scale", 1.0)
    return (alpha / rank) * strength * reference_scale


def lora_delta_matrix(element) -> Tensor:
    """The scaled low-rank update as an [out, in] matrix."""
    weight_a = element.params["lora_down"].view(element.params["lora_down"].size(0), -1)
    weight_b = element.params["lora_up"].view(element.params["lora_up"].size(0), -1)
    return (weight_b @ weight_a) * lora_scale(element)


def has_lora_weight_target(original_module: nn.Module, element) -> bool:
    weight = getattr(original_module, target_param(element), None)
    return isinstance(weight, Tensor) and weight.dim() >= 2


def has_lora_activation_path(original_module: nn.Module, element) -> bool:
    """The activation path computes F.op(x, delta), which is only the weight delta's effect for a known
    op applied to its standard-layout `weight`."""
    if type(original_module) not in LORA_HANDLERS or target_param(element) != "weight":
        return False
    layout = WeightLayout.from_metadata(element.metadata.get("layout"))
    return layout.is_standard(original_module.weight.dim())


@register_kernel("lora")
class LoRAKernel(Kernel):
    """
    Executes standard Low-Rank Adaptation (LoRA).
    The delta is defined in weight space, so the kernel applies to any module whose forward reads the
    targeted parameter. For plain linear/conv layers it also offers an activation-space path, which is
    purely additive and highly optimized for use with the HookInjector.
    """
    traits = {
        "is_additive": True,
        "requires_weight_fusion": False
    }

    def is_supported(self, original_module: nn.Module) -> bool:
        return type(original_module) in LORA_HANDLERS

    def supports_activation(self, original_module: nn.Module, element) -> bool:
        return has_lora_activation_path(original_module, element)

    def supports_weights(self, original_module: nn.Module, element) -> bool:
        return has_lora_weight_target(original_module, element)

    def forward_delta(self, x: Tensor, base_output: Tensor, element, original_module) -> Tensor:
        """Computes the activation delta without triggering original_module."""
        handler = LORA_HANDLERS.get(type(original_module))
        if not handler:
            raise TypeError(f"Unhandled module type: {type(original_module)}")

        # Parameters from Element
        lora_a = element.params["lora_down"].to(x.device, dtype=x.dtype)
        lora_b = element.params["lora_up"].to(x.device, dtype=x.dtype)

        # LoRA Path
        # Using the lambda handlers to normalize the different F signatures
        lx = handler(x, lora_a, None, original_module)
        lx = handler(lx, lora_b, None, original_module)

        return lx * lora_scale(element)

    def __call__(self, x: Tensor, element, original_module):
        base_output = original_module(x)
        return base_output + self.forward_delta(x, base_output, element, original_module)

    def compute_delta(self, element, original_module) -> dict:
        param = target_param(element)
        w_orig = getattr(original_module, param)
        layout = WeightLayout.from_metadata(element.metadata.get("layout"))
        weight_delta = layout.from_matrix(lora_delta_matrix(element), w_orig.shape)
        weight_delta = weight_delta.to(device=w_orig.device, dtype=w_orig.dtype)

        # Return a map of updates
        updates = {
            param: weight_delta
        }

        # Offload the bias check here
        if "bias" in element.params:
            updates["bias"] = element.params["bias"]

        return updates

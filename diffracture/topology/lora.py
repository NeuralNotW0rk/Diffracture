import math

import torch
import torch.nn as nn

from .base_element import Element
from .layout import WeightLayout
from ..registry import register_element


def reference_scale_for(weight: torch.Tensor, layout: WeightLayout) -> float:
    """
    Returns the base weight's RMS relative to the standard 1/sqrt(fan_in) init scale.
    Kernels multiply the low-rank delta by this so its factors mean the same thing on every layer,
    whether the module stores ordinary weights or equalized-learning-rate ones (N(0, 1) raw values
    scaled at runtime, as in StyleGAN).
    """
    _, in_sizes = layout.split_shape(weight.shape)
    rms = weight.detach().float().pow(2).mean().sqrt().item()
    return rms * math.sqrt(math.prod(in_sizes)) if rms > 0.0 else 1.0


@register_element("lora")
class LoRAElement(Element):
    """
    A data container for Low-Rank Adaptation parameters.
    Stored in the Grating and utilized by the LoRAKernel.

    The element targets the `param` tensor of the module at `address` (default "weight"). `layout`
    says how that tensor folds into the [out, in] matrix the delta lives in (see WeightLayout), so any
    module that reads its weight in forward can be adapted, not just the standard torch layers.
    """
    def __init__(
        self,
        address: str,
        rank: int,
        alpha: float,
        in_features: int,
        out_features: int,
        kernel_size: tuple = None,
        param: str = "weight",
        layout: dict = None,
        reference_scale: float = 1.0
    ):
        # Initialize the base class with the address and kernel type
        super().__init__(address, kernel_type="lora")

        # Metadata
        self.metadata.update({
            "rank": rank,
            "alpha": alpha,
            "in_features": in_features,
            "out_features": out_features,
            "kernel_size": kernel_size,
            "param": param,
            "layout": layout,
            "reference_scale": reference_scale
        })

        # Parameter Initialization
        if kernel_size:
            # Convolutional LoRA: A is [rank, in, k], B is [out, rank, 1]
            a_shape = (rank, in_features, *kernel_size)
            b_shape = (out_features, rank, *(1 for _ in kernel_size))
        else:
            # Linear LoRA: A is [rank, in], B is [out, rank]
            a_shape = (rank, in_features)
            b_shape = (out_features, rank)

        # Parameter Storage
        # Initializing A with Kaiming Uniform and B with Zeros
        self.params["lora_down"] = nn.Parameter(torch.empty(a_shape))
        self.params["lora_up"] = nn.Parameter(torch.zeros(b_shape))

        nn.init.kaiming_uniform_(self.params["lora_down"], a=5**0.5)

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            rank=metadata["rank"],
            alpha=metadata["alpha"],
            in_features=metadata["in_features"],
            out_features=metadata["out_features"],
            kernel_size=metadata.get("kernel_size"),
            param=metadata.get("param") or "weight",
            layout=metadata.get("layout"),
            reference_scale=metadata.get("reference_scale", 1.0)
        )

    @classmethod
    def for_parameter(
        cls,
        address: str,
        weight: torch.Tensor,
        rank: int,
        alpha: float = None,
        param: str = "weight",
        layout: WeightLayout = None,
        relative_scale: bool = True
    ) -> "LoRAElement":
        """
        Sizes an element from the tensor it will adapt. The layout is inferred from the shape when
        not given; relative_scale expresses the delta in units of the base weight's own scale.
        """
        if weight.dim() < 2:
            raise ValueError(f"LoRA needs a weight with at least 2 dims; '{address}.{param}' has shape {tuple(weight.shape)}.")
        layout = layout or WeightLayout.infer(weight.shape)
        out_features, in_sizes = layout.split_shape(weight.shape)
        return cls(
            address=address,
            rank=rank,
            alpha=alpha if alpha is not None else float(rank),
            in_features=in_sizes[0] if in_sizes else 1,
            out_features=out_features,
            kernel_size=tuple(in_sizes[1:]) or None,
            param=param,
            layout=layout.to_metadata(),
            reference_scale=reference_scale_for(weight, layout) if relative_scale else 1.0
        )


def find_lora_targets(model: nn.Module, include=None, exclude=None) -> list[tuple[str, str, torch.Tensor]]:
    """
    Lists the (module_address, param_name, tensor) sites a weight-space LoRA can attach to:
    every directly owned parameter with at least 2 dims, whatever the module's type.
    include/exclude are regexes matched against the full parameter name ("a.b.weight").
    """
    import re

    targets = []
    for module_address, module in model.named_modules():
        for param_name, tensor in module.named_parameters(recurse=False):
            if tensor.dim() < 2:
                continue
            full_name = f"{module_address}.{param_name}" if module_address else param_name
            if include is not None and not re.search(include, full_name):
                continue
            if exclude is not None and re.search(exclude, full_name):
                continue
            targets.append((module_address, param_name, tensor))
    return targets

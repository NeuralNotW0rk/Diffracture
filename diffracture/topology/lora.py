import torch
import torch.nn as nn

from .base_element import Element
from ..registry import register_element

@register_element("lora")
class LoRAElement(Element):
    """
    A data container for Low-Rank Adaptation parameters.
    Stored in the Grating and utilized by the LoRAKernel.
    """
    def __init__(self, address: str, rank: int, alpha: float, in_features: int, out_features: int, kernel_size: tuple = None):
        # Initialize the base class with the address and kernel type
        super().__init__(address, kernel_type="lora")
        
        # Metadata
        self.metadata.update({
            "rank": rank,
            "alpha": alpha,
            "in_features": in_features,
            "out_features": out_features,
            "kernel_size": kernel_size
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
            kernel_size=metadata.get("kernel_size")
        )
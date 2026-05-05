import torch
import torch.nn as nn

from .base_prism import Prism
from ..registry import register_prism

@register_prism("dora")
class DoRAPrism(Prism):
    """
    A data container for Weight-Decomposed Low-Rank Adaptation (DoRA) parameters.
    Stored in the Lattice and utilized by the DoRAKernel.
    """
    def __init__(self, address: str, rank: int, alpha: float, in_features: int, out_features: int, kernel_size: tuple = None):
        super().__init__(address, kernel_type="dora")
        
        self.metadata.update({
            "rank": rank,
            "alpha": alpha,
            "in_features": in_features,
            "out_features": out_features,
            "kernel_size": kernel_size
        })

        if kernel_size:
            a_shape = (rank, in_features, *kernel_size)
            b_shape = (out_features, rank, *(1 for _ in kernel_size))
        else:
            a_shape = (rank, in_features)
            b_shape = (out_features, rank)

        self.params["lora_down"] = nn.Parameter(torch.empty(a_shape))
        self.params["lora_up"] = nn.Parameter(torch.zeros(b_shape))
        self.params["magnitude"] = nn.Parameter(torch.ones(out_features))

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
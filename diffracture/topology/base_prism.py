import torch.nn as nn


class Prism(nn.Module):
    def __init__(self, address: str, kernel_type: str):
        super().__init__()
        self.address = address
        self.kernel_type = kernel_type
        self.params = nn.ParameterDict()
        self.metadata = {}
        self.active = True
        self.multiplier = 1.0

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        """Instantiates a Prism from its serialized metadata."""
        raise NotImplementedError("Subclasses must implement 'from_metadata'.")
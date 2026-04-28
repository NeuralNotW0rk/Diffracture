import torch.nn as nn


class Prism(nn.Module):
    def __init__(self, address: str, kernel_type: str):
        super().__init__()
        self.address = address
        self.kernel_type = kernel_type
        self.params = nn.ParameterDict()
        self.metadata = {}
        self.active = True
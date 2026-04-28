import torch.nn as nn

class Lattice(nn.Module):
    """
    A central container for Prisms and Kernels.
    Acts as the 'Source of Truth' for a specific set of model interventions.
    """
    def __init__(self):
        super().__init__()
        # Internal PyTorch storage to ensure .to(device) and .state_dict() work
        self._nodes = nn.ModuleDict()
        
        # Registry for stateless kernels
        self.kernels = {}

    @property
    def nodes(self) -> dict:
        """
        Returns a mapping of original addresses to Prisms.
        Preserves the human-interpretable path format (e.g., 'path/to/module').
        """
        return {prism.address: prism for prism in self._nodes.values()}

    def add_prism(self, prism):
        """
        Registers a Prism in the lattice.
        Automatically handles PyTorch naming constraints.
        """
        # ModuleDict keys cannot contain '/' or '.', so we create a safe internal ID
        safe_id = prism.address.replace('/', '__').replace('.', '__')
        self._nodes[safe_id] = prism

    def add_kernel(self, name: str, kernel):
        """Registers a stateless mathematical kernel."""
        self.kernels[name] = kernel

    def get_kernel(self, name: str):
        """Retrieves a kernel by its string identifier (e.g., 'lora')."""
        if name not in self.kernels:
            raise KeyError(f"Kernel '{name}' not found in Lattice registry.")
        return self.kernels[name]
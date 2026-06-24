import torch
import torch.nn as nn
import json
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path
from ..registry import get_element, get_kernel

class Grating(nn.Module):
    """
    A central container for Elements and Kernels.
    Acts as the 'Source of Truth' for a specific set of model interventions.
    """
    def __init__(self):
        super().__init__()
        # Internal PyTorch storage to ensure .to(device) and .state_dict() work
        # Using ModuleList sidesteps the strict key naming constraints of ModuleDict
        self._nodes = nn.ModuleList()
        
        # Registry for stateless kernels
        self.kernels = {}

    @property
    def nodes(self) -> dict:
        """
        Returns a mapping of original addresses to Elements.
        Uses native PyTorch dot notation (e.g., 'path.to.module').
        """
        return {element.address: element for element in self._nodes}

    def add_element(self, element):
        """
        Registers an Element in the grating.
        """
        self._nodes.append(element)

    def add_kernel(self, name: str, kernel):
        """Registers a stateless mathematical kernel."""
        self.kernels[name] = kernel

    def set_multiplier(self, multiplier: float):
        """Adjusts the inference strength for all active Elements."""
        for element in self._nodes:
            element.multiplier = multiplier

    def get_kernel(self, name: str):
        """Retrieves a kernel by its string identifier (e.g., 'lora')."""
        if name in self.kernels:
            return self.kernels[name]
            
        # Fallback: automatically instantiate stateless kernel from the registry
        try:
            kernel_cls = get_kernel(name)
            self.kernels[name] = kernel_cls()
            return self.kernels[name]
        except KeyError:
            raise KeyError(f"Kernel '{name}' not found in Grating cache or global registry.")

    def save(self, path: str, topology_path: str = None):
        """Saves a Grating to a safetensors file with embedded topology metadata.
        Optionally exports the topology to a separate JSON file for offline editing."""
        topology = []
        for element in self._nodes:
            topology.append({
                "address": element.address,
                "kernel_type": element.kernel_type,
                "metadata": element.metadata
            })

        if topology_path:
            with open(topology_path, "w") as f:
                json.dump(topology, f, indent=4)
                
        state_dict = self.state_dict()
        if len(state_dict) == 0:
            # Safetensors crashes if the state_dict is completely empty.
            # We insert a tiny dummy tensor to allow saving weightless Gratings natively.
            state_dict["__grating_dummy__"] = torch.zeros(1, dtype=torch.int8)
            
        metadata = {"topology": json.dumps(topology)}
        save_file(state_dict, path, metadata=metadata)

    @classmethod
    def load(cls, path: str, topology_path: str = None):
        """Loads a Grating from a safetensors file. 
        If topology_path is provided, it overrides the embedded topology metadata."""
        grating = cls()
        
        topology = None
        if topology_path and Path(topology_path).exists():
            with open(topology_path, "r") as f:
                topology = json.load(f)
        
        with safe_open(path, framework="pt", device="cpu") as f:
            if topology is None:
                if not f.metadata() or "topology" not in f.metadata():
                    raise FileNotFoundError(f"Topology metadata not found in {path}, and no topology_path provided.")
                topology = json.loads(f.metadata()["topology"])
                
            for node in topology:
                ktype = node["kernel_type"]
                element_cls = get_element(ktype)
                element = element_cls.from_metadata(node["address"], node["metadata"])
                grating.add_element(element)

            state_dict = {key: f.get_tensor(key) for key in f.keys() if key != "__grating_dummy__"}
            if state_dict:
                grating.load_state_dict(state_dict, strict=False)
            
        return grating

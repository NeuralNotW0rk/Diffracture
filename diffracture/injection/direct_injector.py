import torch
import torch.nn as nn
from .base_injector import Injector
from ..registry import register_injector

@register_injector("direct")
class DirectInjector(Injector):
    """
    Performs permanent weight modification immediately upon injection.
    No wrappers or hooks are maintained after the operation.
    """
    def __init__(self):
        super().__init__()
        self._fused = False

    def inject(self, target_model, lattice):
        """
        Iterates through the lattice and modifies model weights in-place permanently.
        """
        if self._fused:
            print("Direct injection already performed.")
            return
            
        print("Performing direct weight injection...")
        modified_count = 0
        with torch.no_grad():
            for address, prism in lattice.nodes.items():
                try:
                    original_module = target_model.get_submodule(address)
                except AttributeError:
                    continue

                kernel = lattice.get_kernel(prism.kernel_type)
                deltas = kernel.compute_delta(prism, original_module)
                
                for param_name, delta in deltas.items():
                    if hasattr(original_module, param_name):
                        target_param = getattr(original_module, param_name)
                        target_param.add_(delta)
                
                modified_count += 1

        self._fused = True
        print(f"Directly merged {modified_count} modules into the base weights.")

    def on_inject(self, target_model):
        """
        Direct injection is permanent; usually, no further state 
        management is required after weights are added.
        """
        pass

    def on_extract(self, target_model):
        """
        Extracting from a direct injection is difficult because the
        divergence is now merged with the base. Best to use the Lattice.
        """
        print("Warning: DirectInjector cannot extract divergence from merged weights.")
        return {}

    def on_collapse(self, target_model, lattice):
        """
        No-op for DirectInjector since weights are merged immediately on injection.
        """
        print("Model is already collapsed (DirectInjector is permanent).")

    def cleanup(self, target_model):
        """
        Direct injection cannot be reversed via cleanup as the 
        base weights have been modified in-place.
        """
        print("Note: Cleanup has no effect on DirectInjector (permanent modification).")
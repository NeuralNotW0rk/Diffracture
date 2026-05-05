import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from .base_injector import Injector
from ..registry import register_injector


class UnparametrizedModuleProxy:
    """
    A transparent proxy that intercepts attribute access and returns the 
    base, unparametrized tensors. This prevents infinite recursion when 
    Kernels attempt to read `module.weight` during a parametrization calculation.
    """
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        # If the parameter is parametrized, safely return the base tensor
        if hasattr(self._module, "parametrizations") and name in self._module.parametrizations:
            return self._module.parametrizations[name].original
        
        # Otherwise, pass the request through to the original module
        return getattr(self._module, name)


class DiffractureParametrization(nn.Module):
    def __init__(self, prism, kernel, original_module, param_name):
        super().__init__()
        self.prism = prism
        self.kernel = kernel
        self.param_name = param_name
        self.proxy = UnparametrizedModuleProxy(original_module)

    def forward(self, x):
        # O(1) Short-circuit logic to instantly disable the adapter
        if not self.prism.active or self.prism.multiplier == 0.0:
            return x

        # We pass the safe proxy into the kernel so it can read base weights safely
        deltas = self.kernel.compute_delta(self.prism, self.proxy)
        
        delta = deltas.get(self.param_name)
        if delta is not None:
            return x + delta
            
        return x


@register_injector("parametrize")
class ParametrizeInjector(Injector):
    """
    Uses PyTorch's native parametrization API to modify weights dynamically 
    before the forward pass. Ideal for Weight-Decomposed kernels like DoRA.
    """
    def __init__(self):
        super().__init__()
        self._parametrized_modules = {}

    def inject(self, target_model, lattice):
        for address, prism in lattice.nodes.items():
            try:
                original_module = target_model.get_submodule(address)
            except AttributeError:
                continue

            kernel = lattice.get_kernel(prism.kernel_type)
            if not kernel.is_supported(original_module):
                continue

            # Perform a dry-run to discover which parameters this Kernel updates
            deltas = kernel.compute_delta(prism, original_module)
            param_names = list(deltas.keys())
            
            self._parametrized_modules[address] = {"module": original_module, "params": param_names, "prism": prism}

            for param_name in param_names:
                if hasattr(original_module, param_name):
                    p_wrapper = DiffractureParametrization(prism, kernel, original_module, param_name)
                    parametrize.register_parametrization(original_module, param_name, p_wrapper)

        print(f"Successfully parametrized {len(self._parametrized_modules)} modules.")

    def on_inject(self, target_model):
        # Freeze the original weights so only Prisms train
        for info in self._parametrized_modules.values():
            original_module = info["module"]
            for param in original_module.parameters():
                param.requires_grad = False

    def on_extract(self, target_model) -> dict:
        return {addr: info["prism"].state_dict() for addr, info in self._parametrized_modules.items()}
    
    def on_collapse(self, target_model, lattice):
        # PyTorch has a built in function to perfectly fuse parametrizations
        for info in self._parametrized_modules.values():
            original_module = info["module"]
            for param_name in info["params"]:
                # leave_parametrized=True permanently overwrites the base tensor with the fused output
                parametrize.remove_parametrizations(original_module, param_name, leave_parametrized=True)
        
        self._parametrized_modules.clear()
        print("Parametrizations permanently merged into base weights.")

    def cleanup(self, target_model):
        for info in self._parametrized_modules.values():
            original_module = info["module"]
            for param_name in info["params"]:
                # leave_parametrized=False cleanly unhooks our math and restores original weights
                parametrize.remove_parametrizations(original_module, param_name, leave_parametrized=False)
        
        self._parametrized_modules.clear()
        print("Model restored to original state. Parametrizations cleared.")
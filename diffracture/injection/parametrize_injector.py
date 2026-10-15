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
    def __init__(self, element, kernel, original_module, param_name):
        super().__init__()
        # Held outside the module tree: registering the element as a submodule would make its
        # parameters part of the target module's, so freezing the target would freeze the element too
        object.__setattr__(self, "element", element)
        self.kernel = kernel
        self.param_name = param_name
        self.proxy = UnparametrizedModuleProxy(original_module)

    def delta(self):
        # We pass the safe proxy into the kernel so it can read base weights safely
        return self.kernel.compute_delta(self.element, self.proxy).get(self.param_name)

    def forward(self, x):
        # O(1) Short-circuit logic to instantly disable the adapter
        if not self.element.active or self.element.multiplier == 0.0:
            return x

        delta = self.delta()
        if delta is not None:
            return x + delta

        return x


def _remove_parametrization(module, param_name, p, merge: bool):
    """
    Removes one parametrization from a parameter's chain, leaving any others (e.g. from another
    grating on the same module) in place. With merge, its delta is first added to the base tensor.
    """
    plist = module.parametrizations[param_name]
    if merge:
        with torch.no_grad():
            delta = p.delta()
            if delta is not None:
                plist.original.add_(delta)

    if len(plist) == 1:
        parametrize.remove_parametrizations(module, param_name, leave_parametrized=False)
    else:
        del plist[next(i for i, q in enumerate(plist) if q is p)]


@register_injector("parametrize")
class ParametrizeInjector(Injector):
    """
    Uses PyTorch's native parametrization API to modify weights dynamically
    before the forward pass. Ideal for Weight-Decomposed kernels like DoRA, and for weight-space
    kernels on modules whose forward the kernel doesn't know (e.g. StyleGAN's modulated convs).
    """
    def __init__(self):
        super().__init__()
        self._parametrized_modules = {}

    def inject(self, target_model, grating):
        for address, element in grating.nodes.items():
            try:
                original_module = target_model.get_submodule(address)
            except AttributeError:
                continue

            kernel = grating.get_kernel(element.kernel_type)
            if not kernel.supports_weights(original_module, element):
                continue

            self.parametrize_element(address, original_module, element, kernel)

        print(f"Successfully parametrized {len(self._parametrized_modules)} modules.")

    def parametrize_element(self, address, original_module, element, kernel):
        # Perform a dry-run to discover which parameters this Kernel updates
        deltas = kernel.compute_delta(element, original_module)

        parametrizations = {}
        for param_name in deltas.keys():
            if isinstance(getattr(original_module, param_name, None), torch.Tensor):
                p_wrapper = DiffractureParametrization(element, kernel, original_module, param_name)
                parametrize.register_parametrization(original_module, param_name, p_wrapper)
                parametrizations[param_name] = p_wrapper

        self._parametrized_modules[address] = {"module": original_module, "parametrizations": parametrizations, "element": element}

    def on_inject(self, target_model):
        # Freeze the original weights so only Elements train
        for info in self._parametrized_modules.values():
            original_module = info["module"]
            for param in original_module.parameters():
                param.requires_grad = False

    def on_extract(self, target_model) -> dict:
        return {addr: info["element"].state_dict() for addr, info in self._parametrized_modules.items()}

    def on_collapse(self, target_model, grating):
        # Fold each delta into its base tensor, then drop our parametrization from the chain
        for info in self._parametrized_modules.values():
            for param_name, p in info["parametrizations"].items():
                _remove_parametrization(info["module"], param_name, p, merge=True)

        self._parametrized_modules.clear()
        print("Parametrizations permanently merged into base weights.")

    def cleanup(self, target_model):
        for info in self._parametrized_modules.values():
            for param_name, p in info["parametrizations"].items():
                # Unhooks our math and restores original weights, leaving other gratings' parametrizations intact
                _remove_parametrization(info["module"], param_name, p, merge=False)

        self._parametrized_modules.clear()
        print("Model restored to original state. Parametrizations cleared.")

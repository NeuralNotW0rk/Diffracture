import torch
import torch.nn as nn
from .base_injector import Injector
from .parametrize_injector import ParametrizeInjector
from ..registry import register_injector

@register_injector("hook")
class HookInjector(Injector):
    """
    Applies elements as forward hooks on their target modules. Elements whose kernel has no
    activation-space form for a module (e.g. a weight-space LoRA on a module the kernel doesn't
    know) fall back to a weight parametrization, so one grating can mix both kinds.
    """
    def __init__(self):
        super().__init__()
        # Store handles so we can call .remove() later
        self._handles = {}
        # Keep track of which elements are attached to which addresses
        self._attached_elements = {}
        # Weight-space fallback for elements that can't be applied as a hook
        self._weight_injector = ParametrizeInjector()

    def inject(self, target_model, grating):
        """
        Attaches forward hooks to target layers without swapping them.
        """
        for address, element in grating.nodes.items():
            try:
                original_module = target_model.get_submodule(address)
            except AttributeError:
                print(f"Warning: Could not find module at {address}. Skipping.")
                continue

            kernel = grating.get_kernel(element.kernel_type)
            if not kernel.supports_activation(original_module, element):
                if kernel.supports_weights(original_module, element):
                    self._weight_injector.parametrize_element(address, original_module, element, kernel)
                continue

            # Define the wiretap logic
            def hook_fn(module, input, output, e=element, k=kernel):
                # Instantly bypass the wiretap if disabled
                if not e.active or e.multiplier == 0.0:
                    return output

                # input is a tuple, usually (x,)
                x = input[0]
                # Compute only the delta to avoid an infinite recursion loop
                # and add it directly to the module's pre-computed output
                return output + k.forward_delta(x, output, e, module)

            # Register the hook and store the handle
            handle = original_module.register_forward_hook(hook_fn)
            self._handles[address] = handle
            self._attached_elements[address] = element

        parametrized = len(self._weight_injector._parametrized_modules)
        print(f"Successfully hooked {len(self._handles)} modules" + (f" and parametrized {parametrized}." if parametrized else "."))

    def on_inject(self, target_model):
        """
        Freezes parameters. Unlike Grafting, we don't need to sync train/eval
        mode because the original module is still in the tree.
        """
        for address in self._handles.keys():
            original_module = target_model.get_submodule(address)
            for param in original_module.parameters():
                param.requires_grad = False
        self._weight_injector.on_inject(target_model)

    def on_extract(self, target_model) -> dict:
        """
        Harvests data directly from the elements we tracked during inject.
        """
        extracted = {addr: e.state_dict() for addr, e in self._attached_elements.items()}
        extracted.update(self._weight_injector.on_extract(target_model))
        return extracted

    def on_collapse(self, target_model, grating):
        """
        Math is identical to GraftInjector, but we finish by removing hooks.
        """
        with torch.no_grad():
            for address, element in self._attached_elements.items():
                original_module = target_model.get_submodule(address)
                kernel = grating.get_kernel(element.kernel_type)

                deltas = kernel.compute_delta(element, original_module)
                for param_name, delta in deltas.items():
                    if hasattr(original_module, param_name):
                        getattr(original_module, param_name).add_(delta)

        self._weight_injector.on_collapse(target_model, grating)
        self.cleanup(target_model)

    def cleanup(self, target_model):
        """
        The critical step: Unplug the wiretaps.
        """
        for address, handle in self._handles.items():
            handle.remove()

        self._handles.clear()
        self._attached_elements.clear()
        self._weight_injector.cleanup(target_model)
        print("Hooks removed. Model restored to original state.")

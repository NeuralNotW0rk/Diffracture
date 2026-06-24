import torch
import torch.nn as nn
from .base_injector import Injector
from ..registry import register_injector

@register_injector("hook")
class HookInjector(Injector):
    def __init__(self):
        super().__init__()
        # Store handles so we can call .remove() later
        self._handles = {}
        # Keep track of which elements are attached to which addresses
        self._attached_elements = {}

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
            if not kernel.is_supported(original_module):
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

        print(f"Successfully hooked {len(self._handles)} modules.")

    def on_inject(self, target_model):
        """
        Freezes parameters. Unlike Grafting, we don't need to sync train/eval
        mode because the original module is still in the tree.
        """
        for address in self._handles.keys():
            original_module = target_model.get_submodule(address)
            for param in original_module.parameters():
                param.requires_grad = False

    def on_extract(self, target_model) -> dict:
        """
        Harvests data directly from the elements we tracked during inject.
        """
        return {addr: e.state_dict() for addr, e in self._attached_elements.items()}
    
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

        self.cleanup(target_model)

    def cleanup(self, target_model):
        """
        The critical step: Unplug the wiretaps.
        """
        for address, handle in self._handles.items():
            handle.remove()
            
        self._handles.clear()
        self._attached_elements.clear()
        print("Hooks removed. Model restored to original state.")
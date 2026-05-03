import torch
import torch.nn as nn

from .base_injector import Injector


class GraftedModule(nn.Module):
    def __init__(self, original_module, prism, kernel):
        super().__init__()
        self.original_module = original_module
        self.prism = prism
        self.kernel = kernel

    def forward(self, x):
        # The Kernel uses F.linear or F.conv1d internally
        return self.kernel(x, self.prism, self.original_module)
    

class GraftInjector(Injector):
    def __init__(self):
        super().__init__()
        # Internal registry to track what changed for cleanup
        self._original_modules = {}

    def inject(self, target_model, lattice, registry):
        """
        Replaces target layers with GraftedModules.
        """
        for address, prism in lattice.nodes.items():
            # Locate the target module using PyTorch's path logic
            # Prisms store address as 'path/to/module', but torch uses '.'
            torch_path = address.replace('/', '.')
            
            try:
                original_module = target_model.get_submodule(torch_path)
            except AttributeError:
                print(f"Warning: Could not find module at {address}. Skipping.")
                continue

            if type(original_module) not in registry:
                print(f"Warning: No handler for {type(original_module)} at {address}. Skipping.")
                continue

            # Find the parent so we can reassign the attribute
            parent_path = '.'.join(torch_path.split('.')[:-1])
            leaf_name = torch_path.split('.')[-1]
            parent_module = target_model.get_submodule(parent_path) if parent_path else target_model

            # Backup for cleanup()
            self._original_modules[address] = original_module

            # Swap the module
            # The Kernel is fetched based on the Prism's kernel_type (e.g., 'lora')
            kernel = lattice.get_kernel(prism.kernel_type)
            graft = GraftedModule(original_module, prism, kernel)
            
            setattr(parent_module, leaf_name, graft)

        print(f"Successfully grafted {len(self._original_modules)} modules.")

    def on_inject(self, target_model):
        """
        Standardizes the model state after the modules have been swapped.
        """
        for address, original_module in self._original_modules.items():
            # Freeze the original weights so only new parameters update
            for param in original_module.parameters():
                param.requires_grad = False
                
            # Match the training/eval mode of the graft to the original
            # This ensures things like Dropout/BatchNorm behave correctly
            torch_path = address.replace('/', '.')
            graft = target_model.get_submodule(torch_path)
            graft.train(original_module.training)

    def on_extract(self, target_model) -> dict:
        """
        Harvests the trained parameters from the Prisms inside the grafts.
        """
        extracted_data = {}
        
        for address in self._original_modules.keys():
            torch_path = address.replace('/', '.')
            graft = target_model.get_submodule(torch_path)
            
            # Pull the Prism's state_dict
            extracted_data[address] = graft.prism.state_dict()
        
        return extracted_data
    
    def on_collapse(self, target_model, lattice):
        """
        Generic collapse: Applies all deltas provided by the kernel to the module.
        """
        for address, original_module in self._original_modules.items():
            torch_path = address.replace('/', '.')
            graft = target_model.get_submodule(torch_path)
            
            # Offload logic: Kernel returns a dict of {param_name: delta_tensor}
            # e.g., {"weight": tensor, "bias": tensor}
            deltas = graft.kernel.compute_delta(graft.prism, original_module)
            
            # Blind Application
            with torch.no_grad():
                for param_name, delta in deltas.items():
                    if hasattr(original_module, param_name):
                        target_param = getattr(original_module, param_name)
                        target_param.add_(delta)

        self.cleanup(target_model)

    def cleanup(self, target_model):
        """
        Reverses the surgery by putting the original modules back.
        """
        for address, original_module in self._original_modules.items():
            torch_path = address.replace('/', '.')
            parent_path = '.'.join(torch_path.split('.')[:-1])
            leaf_name = torch_path.split('.')[-1]
            parent_module = target_model.get_submodule(parent_path) if parent_path else target_model
            
            setattr(parent_module, leaf_name, original_module)
            
        self._original_modules.clear()
        print("Model restored to original state.")
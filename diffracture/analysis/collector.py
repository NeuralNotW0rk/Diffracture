import os
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import utils


class ActivationCollector:
    """
    Orchestrates hook injection to capture intermediate module output activations
    across multiple forward runs, compiling them into training sets for Exemplar learning.
    """
    def __init__(self, target_model: nn.Module):
        self.model = target_model
        self.collected_activations = {}
        self._handles = []

    def start_collecting(self, addresses: list):
        """Attaches forward hooks to target layer addresses to capture activations."""
        self.collected_activations = {addr: [] for addr in addresses}
        self._handles = []
        
        for address in addresses:
            try:
                module = self.model.get_submodule(address)
            except AttributeError:
                continue
                
            def make_hook(addr):
                def hook_fn(mod, inp, out):
                    # Detach and clone to prevent memory leaks and graph retention
                    self.collected_activations[addr].append(out.detach().clone().cpu())
                return hook_fn
                
            handle = module.register_forward_hook(make_hook(address))
            self._handles.append(handle)

    def stop_collecting(self):
        """Removes all active collection hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def clear(self):
        """Clears previously collected activations."""
        self.collected_activations = {}

    def compile_in_memory(self, address: str, channel_dim: int) -> list:
        """
        Gathers collected activations for a module and splits them into individual
        feature samples mapping to their channel class index.
        Returns a list of tuples: (activation_sample_tensor, channel_class_index)
        """
        tensors = self.collected_activations.get(address, [])
        if not tensors:
            return []
            
        dataset = []
        for tensor in tensors:
            # Split along the resolved channel dimension
            slices = torch.split(tensor, 1, dim=channel_dim)
            for channel_idx, slice_tensor in enumerate(slices):
                # Split along batch dimension to get individual sample instances
                batch_slices = torch.split(slice_tensor, 1, dim=0)
                for batch_slice in batch_slices:
                    # Remove batch dimension to leave [1, H, W] or [L, 1]
                    sample = batch_slice.squeeze(0)
                    dataset.append((sample, channel_idx))
        return dataset

    def export_to_disk(self, address: str, channel_dim: int, root_dir: str):
        """
        Optional utility to write captured activations to disk in an ImageFolder-compatible
        structure (useful for large scale datasets or external classifiers).
        """
        tensors = self.collected_activations.get(address, [])
        if not tensors:
            return
            
        layer_dir = os.path.join(root_dir, address.replace(".", "_"))
        
        index = 0
        for tensor in tensors:
            slices = torch.split(tensor, 1, dim=channel_dim)
            for channel_idx, slice_tensor in enumerate(slices):
                channel_dir = os.path.join(layer_dir, str(channel_idx))
                os.makedirs(channel_dir, exist_ok=True)
                
                batch_slices = torch.split(slice_tensor, 1, dim=0)
                for batch_slice in batch_slices:
                    sample = batch_slice.squeeze(0)
                    
                    # If 2D image activations, save as png
                    if sample.ndim == 3 and sample.size(0) == 1:
                        img_path = os.path.join(channel_dir, f"{str(index).zfill(5)}.png")
                        utils.save_image(
                            sample,
                            img_path,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1)
                        )
                    else:
                        # Otherwise save as a raw pt tensor
                        pt_path = os.path.join(channel_dir, f"{str(index).zfill(5)}.pt")
                        torch.save(sample, pt_path)
                    
                    index += 1

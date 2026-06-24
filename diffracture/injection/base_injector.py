import torch.nn as nn

from ..topology.grating import Grating


class Injector:
    def inject(self, target_model, grating):
        """Must be implemented by subclasses to perform the actual manipulation."""
        raise NotImplementedError("Subclasses must implement the 'inject' method.")

    def on_inject(self, target_model):
        """Optional: Chores to do after manipulation (e.g., freezing parameters)."""
        pass

    def on_extract(self, target_model):
        """Optional: Logic to harvest weights back into a Grating."""
        pass

    def on_collapse(self, target_model, grating):
        """
        Optional: Permanently merges the Grating into the target_model 
        and removes the injection footprint.
        """
        pass

    def cleanup(self, target_model):
        """Must be implemented to reverse the manipulation (unhook/ungraft)."""
        raise NotImplementedError("Subclasses must implement 'cleanup'.")
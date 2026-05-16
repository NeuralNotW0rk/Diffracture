from .registry import get_injector

class Actant:
    def __init__(self, target_model):
        self.model = target_model
        self.active_injections = []

    def activate(self, lattice, injection_strategy: str = "graft", strength: float = 1.0):
        # Attach strength to the lattice and its prisms so individual kernels can interpret it
        lattice.strength = strength
        if hasattr(lattice, "_nodes"):
            for prism in lattice._nodes:
                prism.strength = strength
                
        injector_cls = get_injector(injection_strategy)
        injector = injector_cls()
        
        # Modification
        injector.inject(self.model, lattice)
        
        # Prepare for divergence
        injector.on_inject(self.model)
        
        self.active_injections.append((injector, lattice))

    def deactivate(self):
        """
        Reverts all active injections, restoring the model to its original state.
        """
        if not self.active_injections:
            return
            
        for injector, lattice in reversed(self.active_injections):
            injector.cleanup(self.model)
                
        self.active_injections.clear()

    def extract(self) -> list:
        """
        Harvests the trained parameters into a portable format.
        """
        if not self.active_injections:
            raise RuntimeError("Actant must be activated before extraction.")
        return [injector.on_extract(self.model) for injector, _ in self.active_injections]

    def collapse_and_cleanup(self):
        """
        Permanently merges weights and removes library footprint.
        """
        if not self.active_injections:
            raise RuntimeError("Actant must be activated before collapse.")
        for injector, lattice in self.active_injections:
            injector.on_collapse(self.model, lattice)
        self.active_injections.clear()
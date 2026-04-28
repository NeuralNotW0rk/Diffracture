
class Actant:
    def __init__(self, target_model, lattice, injector, registry):
        self.model = target_model
        self.lattice = lattice
        self.injector = injector
        self.registry = registry

    def activate(self):
        """
        The standard 'Go' signal for the prototype.
        """
        # Modification
        self.injector.inject(self.model, self.lattice, self.registry)
        
        # Prepare for divergence
        self.injector.on_inject(self.model)

    def extract_divergence(self) -> dict:
        """
        Harvests the trained 'fracture' into a portable format.
        """
        return self.injector.on_extract(self.model)

    def collapse_and_cleanup(self):
        """
        Permanently merges weights and removes library footprint.
        """
        self.injector.on_collapse(self.model, self.lattice)
from .registry import get_injector

class Actant:
    def __init__(self, target_model, lattice):
        self.model = target_model
        self.lattice = lattice
        self.injector = None

    def activate(self, injector_name: str = "graft"):
        injector_cls = get_injector(injector_name)
        self.injector = injector_cls()
        
        # Modification
        self.injector.inject(self.model, self.lattice)
        
        # Prepare for divergence
        self.injector.on_inject(self.model)

    def extract(self) -> dict:
        """
        Harvests the trained parameters into a portable format.
        """
        if self.injector is None:
            raise RuntimeError("Actant must be activated before extraction.")
        return self.injector.on_extract(self.model)

    def collapse_and_cleanup(self):
        """
        Permanently merges weights and removes library footprint.
        """
        if self.injector is None:
            raise RuntimeError("Actant must be activated before collapse.")
        self.injector.on_collapse(self.model, self.lattice)
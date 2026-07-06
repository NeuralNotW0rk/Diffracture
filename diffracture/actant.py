import torch
from .registry import get_injector

class Actant:
    def __init__(self, target_model):
        self.model = target_model
        self.active_injections = []

    def activate(self, grating, injection_strategy: str = "graft", strength: float = 1.0):
        # Attach strength to the grating and its elements so individual kernels can interpret it
        grating.strength = strength
        if hasattr(grating, "_nodes"):
            for element in grating._nodes:
                element.strength = strength
                
        injector_cls = get_injector(injection_strategy)
        injector = injector_cls()
        
        # Modification
        injector.inject(self.model, grating)
        
        # Prepare for divergence
        injector.on_inject(self.model)
        
        self.active_injections.append((injector, grating))

    def deactivate(self):
        """
        Reverts all active injections, restoring the model to its original state.
        """
        if not self.active_injections:
            return
            
        for injector, grating in reversed(self.active_injections):
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
        for injector, grating in self.active_injections:
            injector.on_collapse(self.model, grating)
        self.active_injections.clear()

    def analyze_clusters(self, 
                         grating, 
                         loader, 
                         strategy_name: str, 
                         channel_dims: dict, 
                         num_clusters: dict, 
                         **kwargs):
        """
        Orchestrates unsupervised feature clustering across all BendingElements in the grating.
        Automatically updates the element metadata with the computed cluster_map in-place.
        """
        # Lazy import to avoid loading heavy machine learning modules unless analysis is run
        from .analysis.clustering import FeatureClusteringPipeline
        
        # Initialize pipeline
        pipeline = FeatureClusteringPipeline(self.model, strategy_name=strategy_name)
        
        # Identify addresses requiring clustering
        target_addresses = []
        for address, element in grating.nodes.items():
            cluster = element.metadata.get("cluster")
            if cluster is not None:
                if address in channel_dims and address in num_clusters:
                    target_addresses.append(address)
                else:
                    print(f"Warning: Element at '{address}' targets cluster {cluster}, "
                          f"but lacks required 'channel_dims' or 'num_clusters' config. Skipping.")

        if not target_addresses:
            print("No elements in the grating require cluster analysis.")
            return

        # Start collection hooks
        print(f"Activating collectors for: {target_addresses}")
        pipeline.collector.start_collecting(target_addresses)
        
        # Run forward passes on the loader to compile sample activations
        print("Feeding dataset loader through the model to collect activations...")
        for batch in loader:
            with torch.no_grad():
                if isinstance(batch, (list, tuple)):
                    # Assume batch is (x, y) or (x,) - pass first item as input
                    self.model(batch[0])
                elif isinstance(batch, dict):
                    self.model(**batch)
                else:
                    self.model(batch)
                    
        # Stop collection hooks
        pipeline.collector.stop_collecting()
        
        # Run clustering and update elements in-place
        for address in target_addresses:
            element = grating.nodes[address]
            ch_dim = channel_dims[address]
            k_clusters = num_clusters[address]
            
            cluster_map = pipeline.run_clustering(
                address=address,
                channel_dim=ch_dim,
                num_clusters=k_clusters,
                **kwargs
            )
            
            element.metadata["cluster_map"] = cluster_map
            print(f"Successfully populated cluster_map for element at '{address}'.")
            
        pipeline.collector.clear()
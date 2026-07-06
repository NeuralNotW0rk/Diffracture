import torch
import torch.nn as nn
from diffracture.analysis import FeatureClusteringPipeline
from diffracture.topology.bending import InvertElement
from diffracture.actant import Actant
from diffracture.topology.grating import Grating


def test_cnn_clustering():
    print("\n=== Testing CNN Exemplar Clustering Strategy ===")
    
    class MockConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Output of conv has 8 channels/features
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
            
        def forward(self, x):
            return self.conv(x)

    model = MockConvModel()
    
    # 1. Initialize Pipeline
    pipeline = FeatureClusteringPipeline(model, strategy_name="cnn")
    
    # 2. Collect activations by running forward passes
    pipeline.collector.start_collecting(addresses=["conv"])
    
    print("Running forward passes to collect activation maps...")
    # Generate 5 random samples
    for _ in range(5):
        x = torch.randn(2, 3, 16, 16) # Batch size 2, 3 channels, 16x16 grid
        model(x)
        
    pipeline.collector.stop_collecting()
    
    # 3. Run clustering pipeline
    # We cluster the 8 features of 'conv' (channel dim 1) into 3 groups
    results = pipeline.run_clustering(
        address="conv",
        channel_dim=1,
        num_clusters=3,
        bottleneck=5,
        epochs=3,
        batch_size=4,
        device="cpu",
        depth=1 # CNN model depth
    )
    
    print("Clustering Results:")
    for res in results:
        print(f"  Feature {res['feature_index']} -> Cluster {res['cluster_index']}")
        
    # Assertions
    assert len(results) == 8, f"Expected 8 cluster results, got {len(results)}"
    for res in results:
        assert 0 <= res["cluster_index"] < 3, f"Cluster ID {res['cluster_index']} out of bounds"
    print("[OK] CNN Exemplar Clustering verified successfully!")


def test_transformer_clustering():
    print("\n=== Testing Transformer Exemplar Clustering Strategy ===")
    
    class MockTransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Output represents sequence of tokens (B, L, C) where C = 6 features
            self.linear = nn.Linear(8, 6, bias=False)
            
        def forward(self, x):
            return self.linear(x)

    model = MockTransformerModel()
    
    # 1. Initialize Pipeline
    pipeline = FeatureClusteringPipeline(model, strategy_name="transformer")
    
    # 2. Collect activations by running forward passes
    pipeline.collector.start_collecting(addresses=["linear"])
    
    print("Running forward passes to collect activation sequences...")
    # Generate 5 random samples
    for _ in range(5):
        # Sequence of length 12, input dimensions 8
        x = torch.randn(2, 12, 8) 
        model(x)
        
    pipeline.collector.stop_collecting()
    
    # 3. Run clustering pipeline
    # We cluster the 6 features of 'linear' (channel dim 2) into 2 groups
    results = pipeline.run_clustering(
        address="linear",
        channel_dim=2,
        num_clusters=2,
        bottleneck=4,
        epochs=3,
        batch_size=4,
        device="cpu",
        d_model=8,
        nhead=2,
        num_layers=1
    )
    
    print("Clustering Results:")
    for res in results:
        print(f"  Feature {res['feature_index']} -> Cluster {res['cluster_index']}")
        
    # Assertions
    assert len(results) == 6, f"Expected 6 cluster results, got {len(results)}"
    for res in results:
        assert 0 <= res["cluster_index"] < 2, f"Cluster ID {res['cluster_index']} out of bounds"
    print("[OK] Transformer Exemplar Clustering verified successfully!")


def test_dynamic_cluster_resolution():
    print("\n=== Testing Dynamic Cluster Resolution in Elements & Kernels ===")
    
    class MockLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 6, bias=False)
            nn.init.constant_(self.linear.weight, 1.0)
            
        def forward(self, x):
            return self.linear(x)

    model = MockLinearModel()
    x = torch.ones(1, 2, 4)
    
    base_out = model(x)
    
    cluster_map = [
        {"feature_index": 0, "cluster_index": 0},
        {"feature_index": 1, "cluster_index": 0},
        {"feature_index": 2, "cluster_index": 0},
        {"feature_index": 3, "cluster_index": 1},
        {"feature_index": 4, "cluster_index": 1},
        {"feature_index": 5, "cluster_index": 1},
    ]
    
    invert_el = InvertElement(
        address="linear", 
        indices=None,
        cluster=1, 
        cluster_map=cluster_map
    )
    
    grating = Grating()
    grating.add_element(invert_el)
    
    actant = Actant(model)
    actant.activate(grating, injection_strategy="hook")
    
    hooked_out = model(x)
    actant.deactivate()
    
    expected_out = torch.tensor([[[4.0, 4.0, 4.0, -3.0, -3.0, -3.0]]])
    assert torch.allclose(hooked_out, expected_out), f"Output mismatch: {hooked_out}"
    print("[OK] Cluster indices correctly resolved dynamically and transform applied selectively!")


def test_actant_clustering_orchestration():
    print("\n=== Testing Actant-Level Clustering Orchestration ===")
    
    class MockLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 6, bias=False)
            nn.init.constant_(self.linear.weight, 1.0)
            
        def forward(self, x):
            return self.linear(x)

    model = MockLinearModel()
    
    # Define a Grating with an InvertElement targeting cluster 0
    grating = Grating()
    invert_el = InvertElement(address="linear", cluster=0)
    grating.add_element(invert_el)
    
    # Initialize Actant
    actant = Actant(model)
    
    # Setup dataset loader (a list of mock inputs)
    loader = [torch.randn(2, 5, 4) for _ in range(5)]
    
    # Call high-level orchestration method on Actant
    print("Orchestrating analysis pipeline via Actant...")
    actant.analyze_clusters(
        grating=grating,
        loader=loader,
        strategy_name="transformer",
        channel_dims={"linear": 2},
        num_clusters={"linear": 2},
        bottleneck=4,
        epochs=3,
        batch_size=4,
        device="cpu",
        d_model=8,
        nhead=2,
        num_layers=1
    )
    
    # Verify that the element's metadata was populated in-place
    cluster_map = invert_el.metadata.get("cluster_map")
    assert cluster_map is not None, "cluster_map was not populated in the element metadata!"
    assert len(cluster_map) == 6, f"Expected cluster_map of length 6, got {len(cluster_map)}"
    print("Generated Cluster Map:")
    for item in cluster_map:
        print(f"  Feature {item['feature_index']} -> Cluster {item['cluster_index']}")
        
    # Verify execution of the dynamically configured element
    print("Activating model with resolved clusters...")
    x_test = torch.ones(1, 2, 4) # input sum is 4.0
    actant.activate(grating, injection_strategy="hook")
    hooked_out = model(x_test)
    actant.deactivate()
    
    # Get which features resolved to cluster 0
    resolved_features = [item["feature_index"] for item in cluster_map if item["cluster_index"] == 0]
    
    # Verify that only the resolved features were inverted (value should be -3.0, others 4.0)
    for c in range(6):
        val = hooked_out[0, 0, c].item()
        if c in resolved_features:
            assert abs(val - (-3.0)) < 1e-4, f"Feature {c} in cluster 0 was not inverted! Value: {val}"
        else:
            assert abs(val - 4.0) < 1e-4, f"Feature {c} not in cluster 0 was inverted! Value: {val}"
            
    print("[OK] Actant-level clustering orchestration and activation verified successfully!")


if __name__ == "__main__":
    test_cnn_clustering()
    test_transformer_clustering()
    test_dynamic_cluster_resolution()
    test_actant_clustering_orchestration()
    print("\n=== All Clustering Tests Passed! ===")

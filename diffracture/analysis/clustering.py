import torch
import torch.nn as nn
from torch import Tensor

from ..registry import get_exemplar
from .collector import ActivationCollector


def kmeans_pytorch(X: Tensor, num_clusters: int, max_iters: int = 100, tolerance: float = 1e-4) -> Tensor:
    """
    A pure PyTorch, dependency-free K-Means clustering implementation.
    X: [N, D] data matrix.
    Returns: a 1D tensor [N] containing assigned cluster IDs for each point.
    """
    N, D = X.shape
    device = X.device
    
    # Handle edge case where num_clusters >= N
    if num_clusters >= N:
        return torch.arange(N, device=device) % num_clusters

    # Randomly initialize cluster centers
    perm = torch.randperm(N, device=device)
    centers = X[perm[:num_clusters]].clone()
    
    cluster_ids = torch.zeros(N, dtype=torch.long, device=device)
    
    for i in range(max_iters):
        # Compute euclidean distance from all points to all centers
        # cdist output is [N, num_clusters]
        distances = torch.cdist(X, centers)
        
        # Assign each point to the closest cluster
        new_ids = torch.argmin(distances, dim=1)
        
        # Check convergence: if assignments didn't change, we are done
        if torch.equal(new_ids, cluster_ids) and i > 0:
            break
        cluster_ids = new_ids
        
        # Recompute centers
        new_centers = torch.zeros_like(centers)
        for c in range(num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() > 0:
                new_centers[c] = X[mask].mean(dim=0)
            else:
                # If a cluster becomes empty, assign it to a random data point
                rand_idx = torch.randint(0, N, (1,), device=device)
                new_centers[c] = X[rand_idx].squeeze(0)
                
        # Early exit if centers stabilized
        if torch.norm(new_centers - centers) < tolerance:
            centers = new_centers
            break
        centers = new_centers
        
    return cluster_ids


class FeatureClusteringPipeline:
    """
    The orchestrator that drives feature collection, exemplar learning, 
    embedding extraction, and K-Means clustering for network bending.
    """
    def __init__(self, target_model: nn.Module, strategy_name: str = "cnn"):
        self.model = target_model
        self.strategy_cls = get_exemplar(strategy_name)
        self.strategy = self.strategy_cls()
        self.collector = ActivationCollector(target_model)

    def run_clustering(self, 
                       address: str, 
                       channel_dim: int, 
                       num_clusters: int,
                       bottleneck: int = 10,
                       epochs: int = 10,
                       batch_size: int = 32,
                       lr: float = 1e-3,
                       device: str = "cpu",
                       **kwargs) -> list:
        """
        Runs the unsupervised feature clustering pipeline for a target module.
        Returns: A list of dicts mapping each filter to a cluster:
                 [{"feature_index": 0, "cluster_index": X}, ...]
        """
        # 1. Grab collected activations
        raw_dataset = self.collector.compile_in_memory(address, channel_dim)
        if not raw_dataset:
            raise ValueError(
                f"No collected activations found for layer '{address}'. "
                f"Please run forward passes on the model with the collector active first."
            )
            
        # Determine number of feature classes (channels)
        num_classes = max([label for _, label in raw_dataset]) + 1
        
        # Preprocess features according to strategy assumptions
        preprocessed_dataset = []
        for x_sample, label in raw_dataset:
            x_norm = self.strategy.prepare_input(x_sample)
            preprocessed_dataset.append((x_norm, label))
            
        # 2. Train the Exemplar Classifier
        print(f"Training Exemplar Classifier for '{address}' ({num_classes} classes, {len(preprocessed_dataset)} samples)...")
        classifier = self.strategy.get_model(num_classes=num_classes, bottleneck=bottleneck, **kwargs)
        classifier.to(device)
        classifier.train()
        
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        loader = torch.utils.data.DataLoader(
            preprocessed_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, label_batch in loader:
                x_batch = x_batch.to(device)
                label_batch = label_batch.to(device)
                
                optimizer.zero_grad()
                feat_vec, class_logits = classifier(x_batch)
                loss = criterion(class_logits, label_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")
            
        # 3. Extract averaged bottleneck embeddings for each feature/filter
        print("Extracting feature embeddings...")
        classifier.eval()
        
        # Group normalized samples by their class/channel ID
        samples_by_label = {c: [] for c in range(num_classes)}
        for x_norm, label in preprocessed_dataset:
            samples_by_label[label].append(x_norm)
            
        averaged_embeddings = []
        with torch.no_grad():
            for c in range(num_classes):
                c_samples = samples_by_label[c]
                if not c_samples:
                    averaged_embeddings.append(torch.zeros(bottleneck, device=device))
                    continue
                # Stack samples and compute mean bottleneck feature vector
                batch_x = torch.stack(c_samples).to(device)
                feat_vecs, _ = classifier(batch_x)
                mean_embedding = feat_vecs.mean(dim=0)
                averaged_embeddings.append(mean_embedding)
                
        embeddings_tensor = torch.stack(averaged_embeddings)  # [num_classes, bottleneck]
        
        # 4. Perform K-Means clustering on the average embeddings
        print(f"Clustering {num_classes} features into {num_clusters} groups...")
        cluster_ids = kmeans_pytorch(embeddings_tensor, num_clusters=num_clusters)
        
        # Format outputs
        results = []
        for feature_idx, cluster_idx in enumerate(cluster_ids.tolist()):
            results.append({
                "feature_index": int(feature_idx),
                "cluster_index": int(cluster_idx)
            })
            
        return results

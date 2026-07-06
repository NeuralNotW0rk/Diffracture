from .base_exemplar import ExemplarStrategy
from .collector import ActivationCollector
from .clustering import FeatureClusteringPipeline, kmeans_pytorch

# Force registrations on package load
from .cnn_exemplar import CNNExemplarStrategy
from .transformer_exemplar import TransformerExemplarStrategy

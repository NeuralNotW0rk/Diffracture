import torch
import torch.nn as nn
from torch import Tensor

class ExemplarStrategy:
    """
    Base strategy defining the interface for exemplar classifiers.
    Custom strategies subclass this to support different network types (e.g. CNN vs. Transformer)
    and shape assumptions.
    """
    def get_model(self, num_classes: int, bottleneck: int, **kwargs) -> nn.Module:
        """Returns a classifier model that outputs a tuple: (latent_bottleneck_embeddings, class_logits)."""
        raise NotImplementedError

    def prepare_input(self, x: Tensor) -> Tensor:
        """Preprocesses a single feature's activation tensor (e.g. normalizing/shaping) before feeding it to the model."""
        raise NotImplementedError

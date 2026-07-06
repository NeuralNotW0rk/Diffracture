import math
import torch
import torch.nn as nn
from torch import Tensor

from ..registry import register_exemplar
from .base_exemplar import ExemplarStrategy


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x is expected to be [B, L, d_model]
        return x + self.pe[:, :x.size(1)]


class TransformerExemplarClassifier(nn.Module):
    def __init__(self, num_classes: int, bottleneck: int = 10, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        d_model = (d_model // 2) * 2  # Enforce even dimensions for positional encoding
        self.input_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_bottleneck = nn.Linear(d_model, bottleneck)
        self.fc_out = nn.Linear(bottleneck, num_classes)

    def forward(self, x: Tensor) -> tuple:
        # x is expected to be [B, L, 1]
        x = self.input_projection(x)  # [B, L, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # [B, L, d_model]
        x = x.mean(dim=1)  # Context pooling over length dimension L
        feat_vec = self.fc_bottleneck(x)
        class_logits = self.fc_out(feat_vec)
        return feat_vec, class_logits


@register_exemplar("transformer")
class TransformerExemplarStrategy(ExemplarStrategy):
    def get_model(self, num_classes: int, bottleneck: int, **kwargs) -> nn.Module:
        d_model = kwargs.get("d_model", 32)
        nhead = kwargs.get("nhead", 4)
        num_layers = kwargs.get("num_layers", 2)
        return TransformerExemplarClassifier(
            num_classes=num_classes, bottleneck=bottleneck,
            d_model=d_model, nhead=nhead, num_layers=num_layers
        )

    def prepare_input(self, x: Tensor) -> Tensor:
        # Squeeze out extra dimensions to get a 1D sequence of shape [L]
        x = x.squeeze()
        
        # Guard against zero-dimensional tensor
        if x.ndim == 0:
            x = x.unsqueeze(0)
            
        # Format as [L, 1]
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        elif x.ndim == 2:
            if x.size(1) != 1:
                x = x.transpose(0, 1)
                if x.size(1) != 1:
                    x = x.reshape(-1, 1)
        else:
            x = x.reshape(-1, 1)

        # Apply z-score normalization
        mean = x.mean()
        std = x.std()
        x_norm = (x - mean) / (std + 1e-8)
        return x_norm

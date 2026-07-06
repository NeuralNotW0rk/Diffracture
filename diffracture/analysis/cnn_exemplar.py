import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import collections

from ..registry import register_exemplar
from .base_exemplar import ExemplarStrategy


# Based on ShuffleNetV2 implementation details from network-bending
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int):
        super(InvertedResidual, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError("stride value must be between 1 and 3")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class CNNExemplarClassifier(nn.Module):
    def __init__(self, num_classes: int, bottleneck: int = 10, depth: int = 2):
        super().__init__()
        self.num_layers = depth
        self.bottleneck = bottleneck
        input_channels = 1
        output_channels = 32

        self.first_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        shuffle_layers_dict = collections.OrderedDict()
        for i in range(self.num_layers):
            shuffle_layers_dict[str(i) + "res"] = InvertedResidual(output_channels, output_channels, 1)
            shuffle_layers_dict[str(i) + "maxpool"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.shuffleLayers = nn.Sequential(shuffle_layers_dict)
        self.fc_bottleneck = nn.Linear(output_channels, self.bottleneck)
        self.fc_out = nn.Linear(self.bottleneck, num_classes)

    def forward(self, x: Tensor) -> tuple:
        x = self.first_conv(x)
        x = self.maxpool(x)
        if self.num_layers > 0:
            if x.size(2) >= 3 and x.size(3) >= 3:
                x = self.shuffleLayers(x)
        x = x.mean([2, 3])
        feat_vec = self.fc_bottleneck(x)
        class_logits = self.fc_out(feat_vec)
        return feat_vec, class_logits


@register_exemplar("cnn")
class CNNExemplarStrategy(ExemplarStrategy):
    def get_model(self, num_classes: int, bottleneck: int, **kwargs) -> nn.Module:
        depth = kwargs.get("depth", 2)
        return CNNExemplarClassifier(num_classes=num_classes, bottleneck=bottleneck, depth=depth)

    def prepare_input(self, x: Tensor) -> Tensor:
        # Expect x of shape [H, W] or [1, H, W]. Normalize to [1, H, W].
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            if x.size(0) != 1:
                # If there are batch or channel dims, squeeze them to be [1, H, W]
                x = x.mean(dim=0, keepdim=True)
        
        # Apply z-score normalization
        mean = x.mean()
        std = x.std()
        x_norm = (x - mean) / (std + 1e-8)
        return x_norm

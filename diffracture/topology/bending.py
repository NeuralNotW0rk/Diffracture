import torch
import torch.nn as nn

from .base_element import Element
from ..registry import register_element


class BendingElement(Element):
    """Base Element for network bending transforms."""
    def __init__(self, address: str, kernel_type: str, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type=kernel_type)
        self.metadata.update({
            "indices": list(indices) if indices is not None else [],
            "cluster": cluster,
            "cluster_map": cluster_map
        })


@register_element("erode")
class ErodeElement(BendingElement):
    """
    Element for morphological erosion.
    """
    def __init__(self, address: str, radius: int, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="erode", indices=indices, cluster=cluster, cluster_map=cluster_map)
        self.metadata.update({
            "radius": int(radius)
        })

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            radius=metadata.get("radius", 1),
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("dilate")
class DilateElement(BendingElement):
    """
    Element for morphological dilation.
    """
    def __init__(self, address: str, radius: int, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="dilate", indices=indices, cluster=cluster, cluster_map=cluster_map)
        self.metadata.update({
            "radius": int(radius)
        })

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            radius=metadata.get("radius", 1),
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("translate")
class TranslateElement(BendingElement):
    """
    Element for 2D spatial translation.
    """
    def __init__(self, address: str, offset_x: float, offset_y: float, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="translate", indices=indices, cluster=cluster, cluster_map=cluster_map)
        self.metadata.update({
            "offset_x": float(offset_x),
            "offset_y": float(offset_y)
        })

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            offset_x=metadata.get("offset_x", 0.0),
            offset_y=metadata.get("offset_y", 0.0),
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("scale")
class ScaleElement(BendingElement):
    """
    Element for 2D spatial scaling.
    """
    def __init__(self, address: str, scale_factor: float, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="scale", indices=indices, cluster=cluster, cluster_map=cluster_map)
        self.metadata.update({
            "scale_factor": float(scale_factor)
        })

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            scale_factor=metadata.get("scale_factor", 1.0),
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("rotate")
class RotateElement(BendingElement):
    """
    Element for 2D spatial rotation (angle in degrees, 0 to 360).
    """
    def __init__(self, address: str, angle: float, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="rotate", indices=indices, cluster=cluster, cluster_map=cluster_map)
        self.metadata.update({
            "angle": float(angle)
        })

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            angle=metadata.get("angle", 0.0),
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("resize")
class ResizeElement(BendingElement):
    """
    Element for 2D spatial resizing.
    """
    def __init__(self, address: str, scale_x: float, scale_y: float, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="resize", indices=indices, cluster=cluster, cluster_map=cluster_map)
        self.metadata.update({
            "scale_x": float(scale_x),
            "scale_y": float(scale_y)
        })

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            scale_x=metadata.get("scale_x", 1.0),
            scale_y=metadata.get("scale_y", 1.0),
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("flip-h")
class FlipHorizontalElement(BendingElement):
    """
    Element for horizontal reflection.
    """
    def __init__(self, address: str, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="flip-h", indices=indices, cluster=cluster, cluster_map=cluster_map)

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("flip-v")
class FlipVerticalElement(BendingElement):
    """
    Element for vertical reflection.
    """
    def __init__(self, address: str, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="flip-v", indices=indices, cluster=cluster, cluster_map=cluster_map)

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("invert")
class InvertElement(BendingElement):
    """
    Element for activation value inversion.
    """
    def __init__(self, address: str, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="invert", indices=indices, cluster=cluster, cluster_map=cluster_map)

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("binary-thresh")
class BinaryThresholdElement(BendingElement):
    """
    Element for thresholding activation values.
    """
    def __init__(self, address: str, threshold: float, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="binary-thresh", indices=indices, cluster=cluster, cluster_map=cluster_map)
        self.metadata.update({
            "threshold": float(threshold)
        })

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            threshold=metadata.get("threshold", 0.0),
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("scalar-multiply")
class ScalarMultiplyElement(BendingElement):
    """
    Element for channel scalar multiplication.
    """
    def __init__(self, address: str, factor: float, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="scalar-multiply", indices=indices, cluster=cluster, cluster_map=cluster_map)
        self.metadata.update({
            "factor": float(factor)
        })

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            factor=metadata.get("factor", 1.0),
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )


@register_element("ablate")
class AblateElement(BendingElement):
    """
    Element for zeroing/ablating activations.
    """
    def __init__(self, address: str, indices: list = None, cluster: int = None, cluster_map: list = None):
        super().__init__(address, kernel_type="ablate", indices=indices, cluster=cluster, cluster_map=cluster_map)

    @classmethod
    def from_metadata(cls, address: str, metadata: dict):
        return cls(
            address=address,
            indices=metadata.get("indices"),
            cluster=metadata.get("cluster"),
            cluster_map=metadata.get("cluster_map")
        )
